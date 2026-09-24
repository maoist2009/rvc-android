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
 * Headless entry point so the NPU smoke test can be driven from the shell
 * (`am broadcast -a com.dreamlite.rvc.RUN_SMOKE -n <pkg>/...SmokeReceiver`)
 * without launching an activity and disturbing whatever is on screen. Lives in
 * the debug source set — never shipped in a release build.
 */
class SmokeReceiver : BroadcastReceiver() {

    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action != ACTION) return
        val app = context.applicationContext as android.app.Application
        val pending = goAsync()
        CoroutineScope(SupervisorJob() + Dispatchers.IO).launch {
            try {
                val result = SmokeTest(app).run()
                result.lines.forEach { Log.i(TAG, it) }
                Log.i(TAG, "smoke result ok=${result.ok}")
            } catch (t: Throwable) {
                Log.e(TAG, "smoke receiver threw", t)
            } finally {
                pending.finish()
            }
        }
    }

    companion object {
        private const val TAG = "rvc_smoke"
        const val ACTION = "com.dreamlite.rvc.RUN_SMOKE"
    }
}
