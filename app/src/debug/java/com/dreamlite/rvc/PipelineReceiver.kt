package com.dreamlite.rvc

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.AudioTrack
import android.util.Log
import com.dreamlite.rvc.engine.RvcSession
import com.dreamlite.rvc.engine.RvcState
import com.dreamlite.rvc.engine.RvcStats
import com.dreamlite.rvc.service.RvcLiveService
import java.io.File
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch

/**
 * Headless driver for the pipeline tests — same pattern as SmokeReceiver,
 * so the RTF can be measured from the shell without an activity:
 *
 *   sudo /system/bin/am broadcast -a com.dreamlite.rvc.RUN_PIPELINE \
 *     -n com.dreamlite.rvc.tmp/com.dreamlite.rvc.PipelineReceiver
 *
 *   sudo /system/bin/am broadcast -a com.dreamlite.rvc.RUN_REALTIME \
 *     -n com.dreamlite.rvc.tmp/com.dreamlite.rvc.PipelineReceiver \
 *     --es input test_input.f32 --es output stream_output.f32
 *
 * RUN_REALTIME extras: input/output (file names under filesDir, defaults
 * test_input.f32 / stream_output.f32), mic=true to read the microphone instead
 * of the file, speaker=true to play to the earpiece, null=true to discard the
 * output (pure RTF measurement). Models and the input must already be staged
 * under filesDir. Debug source set only; never shipped.
 *
 * Runs a whole-file conversion through the same RvcSession the UI uses, so
 * the measured RTF is the number the app actually sees.
 */
class PipelineReceiver : BroadcastReceiver() {

    override fun onReceive(context: Context, intent: Intent) {
        when (intent.action) {
            ACTION_PIPELINE -> runPipeline(context)
            ACTION_REALTIME -> runRealtime(context, intent)
            ACTION_CONVERT -> runConvert(context, intent)
            ACTION_MINBUF -> logMinBuffers()
            ACTION_MEASURE -> runMeasure(context, intent)
            ACTION_AUDIOTEST -> runAudioTest()
            ACTION_PACETEST -> runPaceTest(context, intent)
        }
    }

    /**
     * Probes whether a low-latency AudioTrack path exists on this device.
     * Writes 10 ms chunks for one second at each config and logs the
     * getTimestamp()-based presentation delay, plus the allocated buffer.
     */
    private fun runAudioTest() {
        Log.i(TAG, "audiotest start")
        CoroutineScope(SupervisorJob() + Dispatchers.IO).launch {
            val cfgs = listOf(
                Triple("48k/lowlat/10msbuf", 48_000, AudioTrack.PERFORMANCE_MODE_LOW_LATENCY),
                Triple("48k/media/10msbuf", 48_000, AudioTrack.PERFORMANCE_MODE_NONE),
                Triple("40k/lowlat/10msbuf", 40_000, AudioTrack.PERFORMANCE_MODE_LOW_LATENCY),
                Triple("48k/lowlat/25msbuf", 48_000, AudioTrack.PERFORMANCE_MODE_LOW_LATENCY),
            )
            for ((label, sr, perf) in cfgs) {
                try {
                    val frames = sr / 100
                    val t = AudioTrack(
                        AudioAttributes.Builder()
                            .setUsage(AudioAttributes.USAGE_MEDIA)
                            .setContentType(AudioAttributes.CONTENT_TYPE_MUSIC)
                            .build(),
                        AudioFormat.Builder()
                            .setSampleRate(sr)
                            .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                            .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                            .build(),
                        frames * 4,
                        AudioTrack.MODE_STREAM,
                        perf,
                    )
                    if (t.state != AudioTrack.STATE_INITIALIZED) {
                        Log.w(TAG, "audiotest $label: NOT initialized")
                        t.release()
                        continue
                    }
                    t.setVolume(0f)
                    t.play()
                    val buf = FloatArray(frames)
                    val ts = android.media.AudioTimestamp()
                    var written = 0L
                    var sum = 0L
                    var n = 0
                    var max = 0L
                    for (i in 0 until 100) {
                        t.write(buf, 0, frames, AudioTrack.WRITE_BLOCKING)
                        written += frames
                        if (i % 10 == 0 && t.getTimestamp(ts)) {
                            val left = (written - ts.framePosition).coerceAtLeast(0L)
                            val ns = ts.nanoTime + left * 1_000_000_000L / sr - System.nanoTime()
                            val ms = ns / 1_000_000L
                            sum += ms
                            n++
                            if (ms > max) max = ms
                        }
                        Thread.sleep(10)
                    }
                    Log.i(
                        TAG,
                        "audiotest $label: state=ok alloc=${t.bufferSizeInFrames} frames " +
                            "(${t.bufferSizeInFrames * 1000 / sr} ms), " +
                            "present avg=${if (n > 0) sum / n else -1} ms max=$max ms",
                    )
                    t.stop()
                    t.release()
                } catch (t: Throwable) {
                    Log.e(TAG, "audiotest $label threw", t)
                }
            }
            Log.i(TAG, "audiotest done")
        }
    }

    /**
     * Digital end-to-end timing test, no mic and no speaker.
     *
     * Feeds a synthetic signal at exactly real time into the same streaming
     * pipeline the live path uses, and logs, for every emitted block, the wall
     * time at which that block became available. The signal carries a 200 ms
     * silence gap centred on every whole second, so the converted output can
     * be checked for content alignment as well: the gap must come out in the
     * expected block, otherwise the block->input mapping is wrong.
     *
     *   am broadcast -a com.dreamlite.rvc.PACETEST \
     *     -n <pkg>/com.dreamlite.rvc.PipelineReceiver --ei seconds 12 --ei chunk 1600
     *
     * This is the ground truth for the Xposed/virtual-mic case: no summing of
     * per-stage timers, just (input sample time -> converted block delivery).
     */
    private fun runPaceTest(context: Context, intent: Intent) {
        val app = context.applicationContext as DreamLiteApp
        val seconds = intent.getIntExtra("seconds", 12).coerceIn(5, 40)
        val chunk = intent.getIntExtra("chunk", 1600).coerceIn(160, 4000)
        Log.i(TAG, "pacetest: ${seconds}s chunk=$chunk")
        CoroutineScope(SupervisorJob() + Dispatchers.IO).launch {
            try {
                RvcLiveService.start(context)
                val err = app.session.ensureInitialized()
                if (err != null) {
                    Log.e(TAG, "pacetest init err=$err")
                    return@launch
                }
                val sr = 16_000
                val outSr = 40_000
                val total = seconds * sr
                // Voiced-ish signal, 200 ms silence centred on every second.
                val sig = FloatArray(total)
                for (n in 0 until total) {
                    val t = n.toDouble() / sr
                    val frac = t - Math.floor(t)
                    if (frac < 0.1 || frac > 0.9) continue
                    sig[n] = (
                        0.15 * Math.sin(2.0 * Math.PI * 130.0 * t) +
                            0.08 * Math.sin(2.0 * Math.PI * 260.0 * t) +
                            0.05 * Math.sin(2.0 * Math.PI * 390.0 * t)
                        ).toFloat()
                }
                val maxBlocks = seconds * 4 + 16
                val emitNs = LongArray(maxBlocks)
                val outBuf = FloatArray(maxBlocks * 10_000)
                RvcJni.pipelineStreamStart()
                val t0 = android.os.SystemClock.elapsedRealtimeNanos()
                var fed = 0
                var lastBlocks = 0L
                var prevStats: RvcStats? = null
                while (fed < total) {
                    val n = minOf(chunk, total - fed)
                    val due = t0 + fed.toLong() * 1_000_000_000L / sr
                    val before = android.os.SystemClock.elapsedRealtimeNanos()
                    if (due > before) {
                        android.os.SystemClock.sleep((due - before) / 1_000_000L)
                    }
                    val tFeed = android.os.SystemClock.elapsedRealtimeNanos()
                    val out = RvcJni.pipelineStreamFeed(sig.copyOfRange(fed, fed + n))
                    fed += n
                    val tAfter = android.os.SystemClock.elapsedRealtimeNanos()
                    if (out.isNotEmpty()) {
                        val st = RvcStats.parse(RvcJni.pipelineStats())
                        val nb = st?.nBlocks ?: 0L
                        val pv = prevStats
                        fun d(cur: Float, get: (RvcStats) -> Float) =
                            if (pv == null) cur else cur - get(pv)
                        val j = (nb - 1).toInt()
                        if (j in 0 until maxBlocks) {
                            emitNs[j] = tAfter
                            val off = j * 10_000
                            if (off + out.size <= outBuf.size) {
                                System.arraycopy(out, 0, outBuf, off, out.size)
                            }
                            val blockStartNs =
                                t0 + (nb - 1) * 4_000L * 1_000_000_000L / sr
                            Log.i(
                                TAG,
                                "PACE chunk=$chunk n=$nb " +
                                    "feedLag=${(tFeed - due) / 1_000_000}ms " +
                                    "feed=${(tAfter - tFeed) / 1_000_000}ms " +
                                    "age=${(tAfter - blockStartNs) / 1_000_000}ms | " +
                                    "c0=${d(st!!.msC0) { it.msC0 }.toInt()} " +
                                    "gn=${d(st.msGn) { it.msGn }.toInt()} " +
                                    "r=${d(st.msR) { it.msR }.toInt()} " +
                                    "enc=${d(st.msEnc) { it.msEnc }.toInt()} " +
                                    "idx=${d(st.msIndex) { it.msIndex }.toInt()} " +
                                    "f0=${d(st.msF0) { it.msF0 }.toInt()} " +
                                    "syn=${d(st.msSynth) { it.msSynth }.toInt()} " +
                                    "sola=${d(st.msSola) { it.msSola }.toInt()}",
                            )
                        }
                        lastBlocks = nb
                        prevStats = st
                    }
                }
                RvcJni.pipelineStreamEnd()
                // Content alignment: find the output RMS minimum around each
                // expected gap and report which block actually carries it.
                val win = 800
                for (d in 1 until seconds) {
                    val expect = d * outSr
                    val lo = maxOf(0, expect - outSr / 2)
                    val hi = minOf(outBuf.size - win, expect + outSr / 2)
                    var best = Double.MAX_VALUE
                    var bestStart = lo
                    var s = lo
                    while (s <= hi) {
                        var acc = 0.0
                        for (k in 0 until win) {
                            val v = outBuf[s + k].toDouble()
                            acc += v * v
                        }
                        if (acc < best) {
                            best = acc
                            bestStart = s
                        }
                        s += 200
                    }
                    val center = bestStart + win / 2
                    val j = center / 10_000
                    val emit = if (j in 0 until maxBlocks) emitNs[j] else 0L
                    val lat = if (emit > 0) (emit - (t0 + d * 1_000_000_000L)) / 1_000_000 else -1L
                    Log.i(
                        TAG,
                        "DIP t=${d}s expectBlock=${d * 4} gotBlock=$j " +
                            "shift=${(center - expect) * 1000 / outSr}ms " +
                            "lat=$lat ms rms=${Math.sqrt(best / win).toInt()}",
                    )
                }
                Log.i(TAG, "pacetest done blocks=$lastBlocks")
            } catch (t: Throwable) {
                Log.e(TAG, "pacetest threw", t)
            }
        }
    }

    /**
     * Runs a live conversion for a fixed number of seconds, with the speaker
     * muted by default, and lets RvcSession's LAT line report the measured
     * end-to-end latency (capture queue + model + AudioTrack/HAL).
     *
     *   am broadcast -a com.dreamlite.rvc.MEASURE \
     *     -n <pkg>/com.dreamlite.rvc.PipelineReceiver \
     *     --ei seconds 25 [--ez mute true]
     */
    private fun runMeasure(context: Context, intent: Intent) {
        val app = context.applicationContext as DreamLiteApp
        val seconds = intent.getIntExtra("seconds", 25).coerceIn(5, 120)
        val mute = intent.getBooleanExtra("mute", true)
        val speaker = intent.getBooleanExtra("speaker", false)
        val monitor = intent.getBooleanExtra("monitor", true)
        Log.i(TAG, "measure: ${seconds}s mute=$mute speaker=$speaker monitor=$monitor")
        CoroutineScope(SupervisorJob() + Dispatchers.IO).launch {
            try {
                RvcLiveService.start(context)
                val err = app.session.ensureInitialized()
                if (err != null) {
                    Log.e(TAG, "measure init err=$err")
                    return@launch
                }
                app.session.setBlockTrace(true)
                app.session.startLive(
                    monitor = monitor,
                    muteMonitor = mute,
                    forceSpeaker = speaker,
                )
                delay(seconds * 1000L)
                app.session.stop()
                app.session.setBlockTrace(false)
                Log.i(TAG, "measure done")
            } catch (t: Throwable) {
                Log.e(TAG, "measure threw", t)
            }
        }
    }

    /**
     * Logs the device's real audio buffer sizing. Latency accounting needs the
     * allocated counts, not the requested ones: the framework is free to round
     * AudioTrack/AudioRecord buffers up, and on this device that difference is
     * exactly what hides behind the "RTF is fine but the delay is seconds".
     */
    private fun logMinBuffers() {
        val rec16 = AudioRecord.getMinBufferSize(
            16_000, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
        )
        Log.i(TAG, "minbuf rec16=$rec16 B (${rec16 / 32.0} ms)")
        probeTrack("out/media/f32/40k", 40_000, AudioFormat.ENCODING_PCM_FLOAT,
            AudioAttributes.USAGE_MEDIA, AudioTrack.PERFORMANCE_MODE_NONE)
        probeTrack("out/media/f32/48k", 48_000, AudioFormat.ENCODING_PCM_FLOAT,
            AudioAttributes.USAGE_MEDIA, AudioTrack.PERFORMANCE_MODE_NONE)
        probeTrack("out/voice/f32/48k", 48_000, AudioFormat.ENCODING_PCM_FLOAT,
            AudioAttributes.USAGE_VOICE_COMMUNICATION, AudioTrack.PERFORMANCE_MODE_LOW_LATENCY)
        probeTrack("out/media/i16/48k", 48_000, AudioFormat.ENCODING_PCM_16BIT,
            AudioAttributes.USAGE_MEDIA, AudioTrack.PERFORMANCE_MODE_LOW_LATENCY)
    }

    /** Logs the *allocated* buffer, which is what the latency actually is. */
    private fun probeTrack(
        label: String,
        sampleRate: Int,
        encoding: Int,
        usage: Int,
        perf: Int,
    ) {
        try {
            val bytesPerFrame = if (encoding == AudioFormat.ENCODING_PCM_16BIT) 2 else 4
            val minBuf = AudioTrack.getMinBufferSize(
                sampleRate, AudioFormat.CHANNEL_OUT_MONO, encoding
            )
            val blockBytes = sampleRate / 4 * bytesPerFrame   // 250 ms
            val t = AudioTrack(
                AudioAttributes.Builder()
                    .setUsage(usage)
                    .setContentType(AudioAttributes.CONTENT_TYPE_MUSIC)
                    .build(),
                AudioFormat.Builder()
                    .setSampleRate(sampleRate)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .setEncoding(encoding)
                    .build(),
                maxOf(minBuf, blockBytes),
                AudioTrack.MODE_STREAM,
                perf,
            )
            val frames = t.bufferSizeInFrames
            Log.i(
                TAG,
                "minbuf $label: min=$minBuf B, alloc=$frames frames " +
                    "(${"%.0f".format(frames * 1000.0 / sampleRate)} ms)",
            )
            t.release()
        } catch (t: Throwable) {
            Log.e(TAG, "minbuf $label failed", t)
        }
    }

    private fun runPipeline(context: Context) {
        val app = context.applicationContext as DreamLiteApp
        val pending = goAsync()
        CoroutineScope(SupervisorJob() + Dispatchers.IO).launch {
            try {
                RvcLiveService.start(context)
                val err = app.session.ensureInitialized()
                Log.i(TAG, if (err == null) "pipeline init ok" else "pipeline init err=$err")
                if (err == null) {
                    // Streaming path — the same one the UI's live session uses
                    // (pipelineConvert is a whole-file helper that is not wired
                    // up in this build's native code).
                    val inPath = File(context.filesDir, "test_input.f32")
                    if (inPath.isFile) {
                        val raw = inPath.readBytes()
                        val n = raw.size / 4
                        val fb = java.nio.ByteBuffer.wrap(raw)
                            .order(java.nio.ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
                        val x = FloatArray(n).also { fb.get(it) }
                        com.dreamlite.rvc.RvcJni.pipelineStreamStart()
                        var outSamples = 0
                        var i = 0
                        while (i < x.size) {
                            val take = minOf(BLOCK, x.size - i)
                            val blk = x.copyOfRange(i, i + take)
                            val emitted = com.dreamlite.rvc.RvcJni.pipelineStreamFeed(blk)
                            outSamples += emitted.size
                            i += take
                        }
                        val tail = com.dreamlite.rvc.RvcJni.pipelineStreamEnd()
                        outSamples += tail.size
                        Log.i(TAG, "stream convert in=${x.size} out=$outSamples stats=${com.dreamlite.rvc.RvcJni.pipelineStats()}")
                    } else {
                        Log.i(TAG, "no test_input.f32 staged")
                    }
                }
            } catch (t: Throwable) {
                Log.e(TAG, "pipeline receiver threw", t)
            } finally {
                try { RvcLiveService.stop(context) } catch (t: Throwable) { }
                pending.finish()
            }
        }
    }

    private fun runRealtime(context: Context, intent: Intent) {
        val app = context.applicationContext as DreamLiteApp
        val mic = intent.getBooleanExtra("mic", false)
        val input = intent.getStringExtra("input") ?: "test_input.f32"
        val output = intent.getStringExtra("output") ?: "stream_output.f32"
        Log.i(TAG, "realtime: mic=$mic input=$input output=$output")
        val pending = goAsync()
        CoroutineScope(SupervisorJob() + Dispatchers.IO).launch {
            try {
                RvcLiveService.start(context)
                val err = app.session.ensureInitialized()
                if (err != null) {
                    Log.e(TAG, "init err=$err")
                    return@launch
                }
                // Realtime f0 lookahead A/B: 0 = earliest window (low
                // latency), 1 = old behaviour (+100 ms, steadier tails).
                intent.getIntExtra("f0_margin", -1).takeIf { it in 0..1 }?.let { m ->
                    app.session.setF0Margin(m)
                }
                com.dreamlite.rvc.RvcJni.pipelineStreamStart()

                val inPath = File(context.filesDir, input)
                if (!mic && !inPath.isFile) {
                    Log.e(TAG, "missing input: $inPath")
                    return@launch
                }

                if (mic) {
                    app.session.startLive(monitor = false)
                    // the session owns capture + playback; wait for it to end
                    // (stop happens on STOP_REALTIME)
                    Log.i(TAG, "live session started")
                } else {
                    // stream the file in 250 ms blocks, same as the capture path
                    val raw = inPath.readBytes()
                    val n = raw.size / 4
                    val fb = java.nio.ByteBuffer.wrap(raw)
                        .order(java.nio.ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
                    val x = FloatArray(n).also { fb.get(it) }
                    val out = ArrayList<FloatArray>()
                    var i = 0
                    while (i < x.size) {
                        val take = minOf(BLOCK, x.size - i)
                        val block = x.copyOfRange(i, i + take)
                        val emitted = com.dreamlite.rvc.RvcJni.pipelineStreamFeed(block)
                        if (emitted.isNotEmpty()) out.add(emitted)
                        i += take
                    }
                    val tail = com.dreamlite.rvc.RvcJni.pipelineStreamEnd()
                    if (tail.isNotEmpty()) out.add(tail)

                    val merged = FloatArray(out.sumOf { it.size })
                    var off = 0
                    for (b in out) { System.arraycopy(b, 0, merged, off, b.size); off += b.size }
                    val bb = java.nio.ByteBuffer.allocate(merged.size * 4)
                        .order(java.nio.ByteOrder.LITTLE_ENDIAN)
                    bb.asFloatBuffer().put(merged)
                    File(context.filesDir, output).writeBytes(bb.array())
                    Log.i(TAG, "streamed in=${x.size} out=${merged.size}")
                }
            } catch (t: Throwable) {
                Log.e(TAG, "realtime receiver threw", t)
            } finally {
                try { RvcLiveService.stop(context) } catch (t: Throwable) { }
                pending.finish()
            }
        }
    }

    private fun runConvert(context: Context, intent: Intent) {
        val app = context.applicationContext as DreamLiteApp
        if (!converting.compareAndSet(false, true)) {
            Log.w(TAG, "convert already running, ignoring")
            return
        }
        val input = intent.getStringExtra("input") ?: "b8231_in.mp3"
        val outF32 = intent.getStringExtra("output") ?: "convert_out.f32"
        val outWav = intent.getStringExtra("wav") ?: "convert_out.wav"
        Log.i(TAG, "convert: input=$input output=$outF32 wav=$outWav")
        val pending = goAsync()
        CoroutineScope(SupervisorJob() + Dispatchers.IO).launch {
            try {
                // Optional retrieval override for A/B: persisted before
                // ensureInitialized, so the signature check sees the change
                // and reloads the pipeline with the new index_rate.
                intent.getFloatExtra("index_rate", -1f).takeIf { it >= 0f }?.let { rate ->
                    app.settingsRepository.setIndexRate(rate)
                    Log.i(TAG, "index_rate override -> $rate")
                }
                // Pitch A/B: persisted, because ensureInitialized re-applies
                // the stored value after every (re)load and would otherwise
                // reset a purely runtime override back to the default 0.
                val pitch = intent.getFloatExtra("pitch", Float.NaN)
                if (!pitch.isNaN()) {
                    app.settingsRepository.setPitchSemitones(pitch)
                    app.session.setPitch(pitch)
                    Log.i(TAG, "pitch override -> $pitch")
                }
                // Same persist-then-push contract for the three tunables.
                intent.getFloatExtra("rms_mix_rate", -1f).takeIf { it >= 0f }?.let { rate ->
                    app.settingsRepository.setRmsMixRate(rate)
                    app.session.setRmsMixRate(rate)
                    Log.i(TAG, "rms_mix_rate override -> $rate")
                }
                intent.getFloatExtra("protect", -1f).takeIf { it >= 0f }?.let { pr ->
                    app.settingsRepository.setProtect(pr)
                    app.session.setProtect(pr)
                    Log.i(TAG, "protect override -> $pr")
                }
                intent.getIntExtra("filter_radius", -1).takeIf { it >= 1 }?.let { fr ->
                    app.settingsRepository.setFilterRadius(fr)
                    app.session.setFilterRadius(fr)
                    Log.i(TAG, "filter_radius override -> $fr")
                }
                // On Android 15+ a background broadcast cannot start a
                // foreground service (mAllowStartForeground false; the process
                // is then killed mid-conversion). Pre-start it from the shell:
                //   am start-foreground-service -n <pkg>/.service.RvcLiveService
                // and keep it alive for the whole run. If the start was denied
                // we still attempt the conversion; it will be slower and may
                // be reaped, and the shell-side error is visible in logcat.
                RvcLiveService.start(context)
                Log.i(TAG, "fgs running=${fgsRunning(context)}")
                val src = File(context.filesDir, input)
                if (!src.isFile) {
                    Log.e(TAG, "missing input: $src")
                    return@launch
                }

                // exact UI path: AudioDecoder -> session.convertOffline
                val samples = com.dreamlite.rvc.data.AudioDecoder.decodeTo16kMono(src)
                    ?: com.dreamlite.rvc.data.AudioDecoder.readF32(src)
                if (samples == null || samples.isEmpty()) {
                    Log.e(TAG, "decode failed for $input")
                    return@launch
                }
                Log.i(TAG, "decoded in=${samples.size} samples @16k mono")

                val dump = intent.getStringExtra("dump")
                if (dump != null) {
                    val db = java.nio.ByteBuffer.allocate(samples.size * 4)
                        .order(java.nio.ByteOrder.LITTLE_ENDIAN)
                    db.asFloatBuffer().put(samples)
                    File(context.filesDir, dump).writeBytes(db.array())
                }

                val err = app.session.convertOffline(samples)
                if (err != null) {
                    Log.e(TAG, "convert err=$err")
                    return@launch
                }
                val out = app.session.lastOutput ?: run {
                    Log.e(TAG, "no output produced")
                    return@launch
                }
                Log.i(TAG, "converted out=${out.size} stats=${com.dreamlite.rvc.RvcJni.pipelineStats()}")

                val bb = java.nio.ByteBuffer.allocate(out.size * 4)
                    .order(java.nio.ByteOrder.LITTLE_ENDIAN)
                bb.asFloatBuffer().put(out)
                File(context.filesDir, outF32).writeBytes(bb.array())

                val wavPath = app.session.saveLastWav(File(context.filesDir, outWav))
                Log.i(TAG, "wav -> $wavPath")
            } catch (t: Throwable) {
                Log.e(TAG, "convert receiver threw", t)
            } finally {
                converting.set(false)
                try { RvcLiveService.stop(context) } catch (t: Throwable) { }
                pending.finish()
            }
        }
    }

    companion object {
        private const val TAG = "rvc_pipeline_test"
        private const val BLOCK = 4000
        private val converting = java.util.concurrent.atomic.AtomicBoolean(false)
        const val ACTION_PIPELINE = "com.dreamlite.rvc.RUN_PIPELINE"
        const val ACTION_REALTIME = "com.dreamlite.rvc.RUN_REALTIME"
        const val ACTION_CONVERT = "com.dreamlite.rvc.RUN_CONVERT"
        const val ACTION_MINBUF = "com.dreamlite.rvc.MINBUF"
        const val ACTION_MEASURE = "com.dreamlite.rvc.MEASURE"
        const val ACTION_AUDIOTEST = "com.dreamlite.rvc.AUDIOTEST"
        const val ACTION_PACETEST = "com.dreamlite.rvc.PACETEST"

        /** True when RvcLiveService is actually in the foreground, i.e. the
         * process will not be reaped mid-conversion. */
        fun fgsRunning(context: Context): Boolean = try {
            val nm = context.getSystemService(Context.ACTIVITY_SERVICE) as android.app.ActivityManager
            nm.getRunningServices(100).any {
                it.service.className == RvcLiveService::class.java.name && it.foreground
            }
        } catch (t: Throwable) { false }
    }
}
