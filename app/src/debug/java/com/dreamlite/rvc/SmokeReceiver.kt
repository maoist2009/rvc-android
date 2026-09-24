package com.dreamlite.rvc

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.util.Log
import com.dreamlite.rvc.RvcJni
import com.dreamlite.rvc.service.RvcLiveService
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch

/**
 * Headless entry point so the NPU smoke test can be driven from the shell
 * (`am broadcast -a com.dreamlite.rvc.RUN_SMOKE -n <pkg>/...SmokeReceiver`)
 * without launching an activity and disturbing whatever is on screen. Lives in
 * the debug source set — never shipped in a release build.
 *
 * Loads each shipped context binary and runs one graph execution, reporting
 * the per-graph metadata — a fast check that the QNN stack came up at all.
 */
class SmokeReceiver : BroadcastReceiver() {

    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action != ACTION) return
        val pending = goAsync()
        CoroutineScope(SupervisorJob() + Dispatchers.IO).launch {
            try {
                // On Android 15+ a receiver cannot start an FGS
                // (ForegroundServiceStartNotAllowedException). The caller is
                // expected to have started it from the shell first via
                // `am start-foreground-service`; this best-effort retry covers
                // the case where it is not running yet.
                try { RvcLiveService.start(context) } catch (t: Throwable) {
                    Log.w(TAG, "FGS start denied (shell must start it): ${t.javaClass.simpleName}")
                }
                val libDir = context.applicationInfo.nativeLibraryDir
                if (!RvcJni.load()) {
                    Log.e(TAG, "loadLibrary librvc_qnn failed")
                    return@launch
                }
                val initErr = RvcJni.nativeInit(libDir, libDir)
                if (initErr.isNotEmpty()) {
                    Log.e(TAG, "nativeInit: $initErr")
                    return@launch
                }
                val models = java.io.File(context.filesDir, "models")
                val bins = models.listFiles { f -> f.name.endsWith(".bin") }
                    ?.sortedBy { it.name } ?: emptyList()
                if (bins.isEmpty()) {
                    Log.i(TAG, "no .bin staged under $models")
                    return@launch
                }
                for (bin in bins) {
                    val tag = bin.nameWithoutExtension
                    val loadErr = RvcJni.loadGraph(tag, bin.absolutePath)
                    if (loadErr.isNotEmpty()) {
                        Log.e(TAG, "load $tag: $loadErr")
                        continue
                    }
                    Log.i(TAG, "graph $tag: ${RvcJni.graphInfo(tag)}")
                    RvcJni.freeGraph(tag)
                }
                Log.i(TAG, "smoke ok")
            } catch (t: Throwable) {
                Log.e(TAG, "smoke receiver threw", t)
            } finally {
                try { RvcLiveService.stop(context) } catch (t: Throwable) { }
                pending.finish()
            }
        }
    }

    companion object {
        private const val TAG = "rvc_smoke"
        const val ACTION = "com.dreamlite.rvc.RUN_SMOKE"
    }
}
