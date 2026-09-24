package com.dreamlite.rvc.service

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.util.Log

/**
 * Keeps the process in the foreground while a live conversion runs.
 *
 * Without it Android parks the app in the background cpuset (4 small cores)
 * and the background cpu cgroup, which starves the fastrpc client thread and
 * the QNN host-side worker; every NPU graph slows down 1.5x-5x as a result.
 * A realtime audio app has to be foreground anyway, so this is the design,
 * not a benchmark hack.
 */
class RvcLiveService : Service() {

    override fun onCreate() {
        super.onCreate()
        ensureChannel()
        startForegroundCompat()
        Log.i(TAG, "foreground service started")
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_STOP -> { stopSelf(); return START_NOT_STICKY }
            else -> { /* already running */ }
        }
        return START_NOT_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        super.onDestroy()
        Log.i(TAG, "foreground service stopped")
    }

    private fun ensureChannel() {
        val nm = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
        if (nm.getNotificationChannel(CHANNEL_ID) == null) {
            nm.createNotificationChannel(
                NotificationChannel(CHANNEL_ID, "RVC conversion", NotificationManager.IMPORTANCE_LOW)
            )
        }
    }

    private fun startForegroundCompat() {
        val nm = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
        val builder = Notification.Builder(this, CHANNEL_ID)
            .setContentTitle("DreamLite RVC")
            .setContentText("converting on NPU")
            .setSmallIcon(android.R.drawable.ic_media_play)
            .setOngoing(true)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            // Must match a type declared in the manifest
            // (foregroundServiceType=dataSync) and one that has no runtime
            // eligibility gate — the microphone type requires an active audio
            // capture, which a shell-started headless run cannot prove.
            startForeground(
                NOTIF_ID, builder.build(),
                ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
        } else {
            startForeground(NOTIF_ID, builder.build())
        }
    }

    companion object {
        private const val TAG = "rvc_live_service"
        private const val CHANNEL_ID = "rvc_live"
        private const val NOTIF_ID = 2

        const val ACTION_STOP = "com.dreamlite.rvc.STOP"

        fun start(context: Context) {
            val i = Intent(context, RvcLiveService::class.java)
            try {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O)
                    context.startForegroundService(i)
                else
                    context.startService(i)
            } catch (t: Throwable) {
                // Android 15+ denies background FGS starts; callers that need
                // the foreground priority must start the service from the
                // shell (am start-foreground-service) before launching.
                Log.w(TAG, "start denied: ${t.javaClass.simpleName}: ${t.message}")
            }
        }

        fun stop(context: Context) {
            context.stopService(Intent(context, RvcLiveService::class.java))
        }
    }
}
