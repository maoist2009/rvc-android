package com.dreamlite.rvc

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.util.Log
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
 */
class PipelineReceiver : BroadcastReceiver() {

    override fun onReceive(context: Context, intent: Intent) {
        when (intent.action) {
            ACTION_PIPELINE -> runPipeline(context)
            ACTION_REALTIME -> runRealtime(context, intent)
        }
    }

    private fun runPipeline(context: Context) {
        val app = context.applicationContext as android.app.Application
        val pending = goAsync()
        CoroutineScope(SupervisorJob() + Dispatchers.IO).launch {
            try {
                // foreground so the process keeps the full cpuset while the
                // graphs run; background scheduling costs 1.5x-5x on every graph
                RvcComputeService.start(context)
                val result = PipelineTest(app).run()
                result.lines.forEach { Log.i(TAG, it) }
                Log.i(TAG, "pipeline result ok=${result.ok}")
            } catch (t: Throwable) {
                Log.e(TAG, "pipeline receiver threw", t)
            } finally {
                try { RvcComputeService.stop(context) } catch (t: Throwable) { }
                pending.finish()
            }
        }
    }

    private fun runRealtime(context: Context, intent: Intent) {
        val app = context.applicationContext as android.app.Application
        /* boolean extras default to false, so absent == the file path */
        val mic = intent.getBooleanExtra("mic", false)
        val speaker = intent.getBooleanExtra("speaker", false)
        val nullSink = intent.getBooleanExtra("null", false)
        val source = when {
            mic -> RealtimeEngine.Source.Mic
            else -> RealtimeEngine.Source.FromFile(
                intent.getStringExtra("input") ?: "test_input.f32")
        }
        val sink = when {
            nullSink -> RealtimeEngine.Sink.Null
            speaker -> RealtimeEngine.Sink.Speaker
            else -> RealtimeEngine.Sink.ToFile(
                intent.getStringExtra("output") ?: "stream_output.f32")
        }
        Log.i(TAG, "realtime: source=$source sink=$sink")
        val pending = goAsync()
        CoroutineScope(SupervisorJob() + Dispatchers.IO).launch {
            try {
                RvcComputeService.start(context)
                val result = RealtimeEngine(app).run(source, sink)
                result.lines.forEach { Log.i(TAG, it) }
                Log.i(TAG, "realtime result ok=${result.ok}")
            } catch (t: Throwable) {
                Log.e(TAG, "realtime receiver threw", t)
            } finally {
                try { RvcComputeService.stop(context) } catch (t: Throwable) { }
                pending.finish()
            }
        }
    }

    companion object {
        private const val TAG = "rvc_pipeline_test"
        const val ACTION_PIPELINE = "com.dreamlite.rvc.RUN_PIPELINE"
        const val ACTION_REALTIME = "com.dreamlite.rvc.RUN_REALTIME"
    }
}
