package com.dreamlite.rvc

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.util.Log
import com.dreamlite.rvc.engine.RvcSession
import com.dreamlite.rvc.engine.RvcState
import com.dreamlite.rvc.service.RvcLiveService
import java.io.File
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
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
                RvcLiveService.start(context)
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
    }
}
