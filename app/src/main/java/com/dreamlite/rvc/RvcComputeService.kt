package com.dreamlite.rvc

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Build
import android.util.Log

/**
 * Keeps the process in the foreground while a conversion is running.
 *
 * Without it Android parks the app in the background cpuset (4 small cores)
 * and the background cpu cgroup, which starves the fastrpc client thread and
 * the QNN host-side worker; every NPU graph slows down 1.5x-5x as a result.
 * A realtime audio app has to be foreground anyway, so this is the design,
 * not a benchmark hack.
 *
 * Notification visibility depends on POST_NOTIFICATIONS being granted; the
 * foreground priority itself does not.
 */
class RvcComputeService : Service() {

    override fun onCreate() {
        super.onCreate()
        val nm = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O &&
            nm.getNotificationChannel(CHANNEL_ID) == null
        ) {
            nm.createNotificationChannel(
                NotificationChannel(CHANNEL_ID, "RVC conversion", NotificationManager.IMPORTANCE_LOW)
            )
        }
        startForeground(NOTIF_ID, buildNotification())
        Log.i(TAG, "foreground service started")
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int = START_NOT_STICKY

    override fun onBind(intent: Intent?) = null

    private fun buildNotification(): Notification {
        val builder = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O)
            Notification.Builder(this, CHANNEL_ID)
        else
            @Suppress("DEPRECATION") Notification.Builder(this)
        return builder
            .setContentTitle("DreamLite RVC")
            .setContentText("converting on NPU")
            .setSmallIcon(android.R.drawable.ic_media_play)
            .setOngoing(true)
            .build()
    }

    companion object {
        private const val TAG = "rvc_service"
        private const val CHANNEL_ID = "rvc_compute"
        private const val NOTIF_ID = 1

        fun start(context: Context) {
            val i = Intent(context, RvcComputeService::class.java)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O)
                context.startForegroundService(i)
            else
                context.startService(i)
        }

        fun stop(context: Context) {
            context.stopService(Intent(context, RvcComputeService::class.java))
        }
    }
}
