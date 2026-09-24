package com.dreamlite.rvc

import android.app.Application
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioRecord
import android.media.AudioTrack
import android.media.MediaRecorder
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.RandomAccessFile
import java.util.concurrent.ArrayBlockingQueue

/**
 * The realtime pipeline wired to a live audio path: a [Source] feeds 16 kHz
 * mono frames into the native streaming pipeline, every emitted 40 kHz block
 * goes to a [Sink].
 *
 * The file→file path is the headless RTF harness — it exercises exactly the
 * same incremental code path as the microphone, but with a deterministic
 * input, so the host can compare streaming RTF and fidelity against the
 * whole-utterance [PipelineTest].
 *
 * Input is read on a small producer thread into a bounded queue and drained
 * in batches.  The native pipeline keeps a block of look-ahead, and the
 * batch is what feeds it: the NPU worker thread only stays busy block to
 * block if the next block's rmvpe windows are already queued, which needs
 * roughly one more chunk staged than the block being emitted.
 *
 * The first ~0.5 s of input is buffered (the encoder context's look-ahead)
 * before any audio is emitted; after that one block comes out per 250 ms of
 * input.  Because a block takes less than 250 ms to compute (RTF < 1), the
 * pipeline waits between blocks in the file mode; with the microphone the
 * input queue paces it for free.
 */
class RealtimeEngine(private val app: Application) {

    sealed class Source {
        /** Read [filesDir]/[name] as raw little-endian float32. */
        class FromFile(val name: String) : Source()
        /** The device microphone, 16 kHz mono. Needs RECORD_AUDIO. */
        object Mic : Source()
    }

    sealed class Sink {
        class ToFile(val name: String) : Sink()
        object Speaker : Sink()
        /** Discard — RTF measurement without audio output. */
        object Null : Sink()
    }

    data class Result(val ok: Boolean, val lines: List<String>)

    suspend fun run(source: Source, sink: Sink): Result = withContext(Dispatchers.IO) {
        val lines = mutableListOf<String>()
        /* logcat's ring buffer is rotated out by the device's audit spam before
         * a headless run can be read back, so the lines also land in filesDir. */
        val logFile = java.io.FileWriter(File(app.filesDir, "realtime_log.txt"))
        fun log(msg: String) {
            Log.i(TAG, msg)
            lines.add(msg)
            logFile.write(msg + "\n"); logFile.flush()
        }
        var record: AudioRecord? = null
        var track: AudioTrack? = null
        try {
            val modelsDir = File(app.filesDir, "models")
            val voiceJson = File(modelsDir, "voice.json")
            if (!modelsDir.exists() || !voiceJson.exists()) {
                return@withContext Result(false, lines +
                    "missing models (expected ${modelsDir.absolutePath})")
            }

            RvcJni.pipelineCreate().takeIf { it.isNotEmpty() }?.let {
                return@withContext Result(false, lines + "pipelineCreate: $it")
            }
            val initErr = RvcJni.pipelineInit(
                modelsDir.absolutePath,
                app.applicationInfo.nativeLibraryDir,
                voiceJson.readText()
            )
            if (initErr.isNotEmpty()) {
                return@withContext Result(false, lines + "pipelineInit: $initErr")
            }
            RvcJni.pipelineStreamStart().takeIf { it.isNotEmpty() }?.let {
                return@withContext Result(false, lines + "streamStart: $it")
            }
            log("stream started: source=$source sink=$sink")

            if (sink is Sink.Speaker) {
                val minBuf = AudioTrack.getMinBufferSize(
                    40_000, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_FLOAT
                )
                track = AudioTrack(
                    android.media.AudioAttributes.Builder()
                        .setUsage(android.media.AudioAttributes.USAGE_VOICE_COMMUNICATION)
                        .setContentType(android.media.AudioAttributes.CONTENT_TYPE_SPEECH)
                        .build(),
                    AudioFormat.Builder()
                        .setSampleRate(40_000)
                        .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                        .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                        .build(),
                    maxOf(minBuf, 10000 * 4 * 2),
                    AudioTrack.MODE_STREAM, AudioManager.AUDIO_SESSION_ID_GENERATE
                )
                if (track?.state != AudioTrack.STATE_INITIALIZED) {
                    return@withContext Result(false, lines + "AudioTrack init failed")
                }
                track!!.play()
            }

            val outCollector = if (sink is Sink.ToFile) ArrayList<Float>() else null
            var nIn = 0
            var nOut = 0
            val feedUs = ArrayList<Long>()      /* per-feed native wall time */
            var emptyFeeds = 0
            val tStart = System.nanoTime()

            if (source is Source.Mic && app.checkSelfPermission(
                    android.Manifest.permission.RECORD_AUDIO) !=
                android.content.pm.PackageManager.PERMISSION_GRANTED
            ) {
                return@withContext Result(false, lines + "RECORD_AUDIO not granted")
            }
            if (source is Source.Mic) {
                val minBuf = AudioRecord.getMinBufferSize(
                    16_000, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_FLOAT
                )
                record = AudioRecord(
                    MediaRecorder.AudioSource.VOICE_COMMUNICATION, 16_000,
                    AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_FLOAT,
                    maxOf(minBuf, 4000 * 4)
                )
                if (record?.state != AudioRecord.STATE_INITIALIZED) {
                    return@withContext Result(false, lines + "AudioRecord init failed")
                }
                record!!.startRecording()
            }

            /* ArrayBlockingQueue rejects null, so the EOF sentinel is an
             * empty chunk — every real chunk is non-empty. */
            val EOF = FloatArray(0)
            val queue = ArrayBlockingQueue<FloatArray>(5)
            val producer = Thread({
                when (source) {
                    is Source.FromFile -> {
                        val input = File(app.filesDir, source.name)
                        if (!input.exists()) { queue.put(EOF); return@Thread }
                        val nTotal = (input.length() / 4).toInt()
                        log("input: $nTotal samples (${nTotal / 16_000.0}s)")
                        val raf = RandomAccessFile(input, "r")
                        val bb = java.nio.ByteBuffer.allocate(4000 * 4)
                            .order(java.nio.ByteOrder.LITTLE_ENDIAN)
                        var pos = 0
                        while (pos < nTotal) {
                            val want = minOf(4000, nTotal - pos)
                            bb.clear()
                            raf.channel.read(bb)
                            bb.flip()
                            val chunk = FloatArray(want)
                            bb.asFloatBuffer().get(chunk, 0, want)
                            pos += want
                            queue.put(chunk)
                        }
                        raf.close()
                    }
                    is Source.Mic -> {
                        val buf = FloatArray(4000)
                        while (running) {
                            val n = record!!.read(buf, 0, 4000, AudioRecord.READ_BLOCKING)
                            if (n <= 0) { log("AudioRecord read returned $n"); break }
                            queue.put(if (n == 4000) buf else buf.copyOf(n))
                        }
                    }
                }
                queue.put(EOF)
            }, "rvc-input")
            producer.start()

            /* drain up to three chunks per feed: one for the block being
             * emitted, the rest is the look-ahead the NPU worker needs. */
            val stage = FloatArray(12000)
            val rest = ArrayList<FloatArray>()
            while (running) {
                val first = queue.take()
                if (first.isEmpty()) break    /* EOF */
                var n = first.size
                System.arraycopy(first, 0, stage, 0, n)
                rest.clear()
                queue.drainTo(rest, 2)
                var eof = false
                for (c in rest) {
                    if (c.isEmpty()) { eof = true; break }
                    System.arraycopy(c, 0, stage, n, c.size)
                    n += c.size
                }
                nIn += n
                val t0 = System.nanoTime()
                val out = RvcJni.pipelineStreamFeed(if (n == stage.size) stage else stage.copyOf(n))
                feedUs.add((System.nanoTime() - t0) / 1000)
                if (out.isEmpty()) {
                    if (emptyFeeds < 3) {   /* state dump: which gate stalled */
                        log("empty feed #$emptyFeeds nIn=$nIn " +
                            "err='${RvcJni.pipelineLastError()}' " +
                            "stats=${RvcJni.pipelineStats()}")
                    }
                    emptyFeeds++
                    if (nIn >= 6720) {
                        val err = RvcJni.pipelineLastError()
                        if (err.isNotEmpty()) {
                            return@withContext Result(false, lines +
                                "streamFeed failed: $err")
                        }
                    }
                }
                nOut += out.size
                emit(out, sink, outCollector, track)
                if (eof) break
            }
            try { producer.interrupt() } catch (t: Throwable) { }

            /* flush the final block: zero-pad the look-ahead window */
            val tail = RvcJni.pipelineStreamEnd()
            nOut += tail.size
            emit(tail, sink, outCollector, track)

            val wallMs = (System.nanoTime() - tStart) / 1_000_000
            val outDurMs = nOut / 40_000.0 * 1000.0
            log("stream done in ${wallMs}ms: $nIn in -> $nOut out " +
                "(${outDurMs / 1000.0}s @40k)")
            log("end-to-end RTF: ${"%.3f".format(wallMs / outDurMs)} " +
                "(incl. reading, sleeping and the ${6720 / 16_000.0}s priming)")
            log("stats: ${RvcJni.pipelineStats()}")
            log("graph_timings: ${RvcJni.graphTimings()}")

            /* per-feed cost: the steady-state block time is what RTF < 1
             * actually depends on, and the worst feed is the one that stalls. */
            feedUs.sort()
            val n2 = feedUs.size
            if (n2 > 0) {
                log("feed wall: n=$n2 median=${feedUs[n2 / 2]}us " +
                    "p90=${feedUs[(n2 * 9 / 10)]}us max=${feedUs.last()}us")
            }

            outCollector?.let {
                val outFile = File(app.filesDir, (sink as Sink.ToFile).name)
                writeFloat32LE(outFile, it.toFloatArray())
                log("wrote ${outFile.absolutePath} (${outFile.length()} bytes)")
                var peak = 0f
                for (v in it) { val a = Math.abs(v); if (a > peak) peak = a }
                log("out peak=$peak samples=${it.size}")
                if (peak < 1e-4f) {
                    return@withContext Result(false, lines + "OUTPUT DEGENERATE (near-silent)")
                }
            }
            Result(true, lines)
        } catch (t: Throwable) {
            log("exception: ${t.javaClass.name}: ${t.message}")
            Log.e(TAG, "realtime engine threw", t)
            Result(false, lines + "exception: ${t.javaClass.name}: ${t.message}")
        } finally {
            try { logFile.close() } catch (t: Throwable) { }
            try { record?.stop() } catch (t: Throwable) { }
            try { record?.release() } catch (t: Throwable) { }
            try { track?.stop(); track?.release() } catch (t: Throwable) { }
            try { RvcJni.pipelineDestroy() } catch (t: Throwable) { }
            running = true
        }
    }

    private fun emit(out: FloatArray, sink: Sink,
                     collector: ArrayList<Float>?, track: AudioTrack?) {
        if (out.isEmpty()) return
        when (sink) {
            is Sink.ToFile -> collector?.addAll(out.toList())
            Sink.Speaker -> {
                if (track != null) track.write(out, 0, out.size,
                    AudioTrack.WRITE_BLOCKING)
            }
            Sink.Null -> { }
        }
    }

    @Volatile private var running = true

    fun stop() { running = false }

    companion object {
        private const val TAG = "rvc_realtime"
    }
}

private fun writeFloat32LE(f: File, x: FloatArray) {
    val bb = java.nio.ByteBuffer.allocate(x.size * 4)
        .order(java.nio.ByteOrder.LITTLE_ENDIAN)
    bb.asFloatBuffer().put(x)
    java.io.FileOutputStream(f).use { it.write(bb.array()) }
}
