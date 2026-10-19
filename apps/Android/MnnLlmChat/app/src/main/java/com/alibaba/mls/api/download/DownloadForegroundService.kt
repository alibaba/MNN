package com.alibaba.mls.api.download

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.util.Log
import com.alibaba.mnnllm.android.R as AppR
import com.alibaba.mnnllm.android.download.DownloadForegroundServiceManager
import com.alibaba.mnnllm.android.main.MainActivity

class DownloadForegroundService : Service() {
    private lateinit var notificationManager: NotificationManager
    private var currentDownloadCount = 0
    private var currentModelName: String? = null
    private var foregroundEntered = false

    override fun onCreate() {
        super.onCreate()
        notificationManager = getSystemService(NotificationManager::class.java)
        createNotificationChannel()
        // A stopService() queued between startForegroundService() and startForeground() can
        // prevent onStartCommand() from ever being delivered, so enter the foreground here.
        enterForeground()
        Log.d(TAG, "onCreate")
    }

    private fun createNotificationChannel() {
        val channel = NotificationChannel(
            CHANNEL_ID,
            getString(AppR.string.download_service_title),
            NotificationManager.IMPORTANCE_LOW
        )
        channel.enableLights(false)
        channel.enableVibration(false)
        channel.setSound(null, null)
        notificationManager.createNotificationChannel(channel)
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        // Extract download count and model name from intent if available
        currentDownloadCount = intent?.getIntExtra(EXTRA_DOWNLOAD_COUNT, 0) ?: 0
        currentModelName = intent?.getStringExtra(EXTRA_MODEL_NAME)
        Log.d(
            TAG,
            "onStartCommand startId=$startId flags=$flags count=$currentDownloadCount modelName=$currentModelName"
        )

        // Never return from here without the service being in the foreground.
        val enteredForeground = enterForeground()
        if (intent == null) {
            Log.w(TAG, "onStartCommand received null intent startId=$startId")
        }
        if (intent == null || !enteredForeground) {
            stopSelf(startId)
        }
        return START_NOT_STICKY
    }

    /** Enters the foreground on the first call and updates the notification afterwards. */
    private fun enterForeground(): Boolean {
        val notification = createNotification()
        if (foregroundEntered) {
            notificationManager.notify(SERVICE_ID, notification)
            return true
        }
        return try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                startForeground(SERVICE_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
            } else {
                startForeground(SERVICE_ID, notification)
            }
            foregroundEntered = true
            // Published only now, so callers can read it as "the foreground state exists".
            instance = this
            Log.d(TAG, "startForeground succeeded")
            DownloadForegroundServiceManager.onServiceReady()
            true
        } catch (e: Exception) {
            Log.e(
                TAG,
                "startForeground failed count=$currentDownloadCount modelName=$currentModelName",
                e
            )
            false
        }
    }

    private fun createNotification(): Notification {
        val contentTitle = getString(AppR.string.download_service_title)
        val contentText = when {
            currentDownloadCount <= 0 -> getString(AppR.string.downloading_please_wait)
            currentDownloadCount == 1 && currentModelName != null -> {
                getString(AppR.string.downloading_single_model, currentModelName)
            }
            else -> {
                getString(AppR.string.downloading_multiple_models, currentDownloadCount)
            }
        }

        // Create intent to open MainActivity and select ModelMarketFragment
        val intent = Intent(this, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP
            putExtra(MainActivity.EXTRA_SELECT_TAB, MainActivity.TAB_MODEL_MARKET)
        }
        
        val pendingIntent = PendingIntent.getActivity(
            this,
            0,
            intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        val builder = Notification.Builder(this, CHANNEL_ID)
            .setContentTitle(contentTitle)
            .setContentText(contentText)
            // Use android system download icon if available, or a fallback
            .setSmallIcon(android.R.drawable.stat_sys_download)
            .setContentIntent(pendingIntent)
            .setAutoCancel(false)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            
        return builder.build()
    }

    fun updateNotification(downloadCount: Int, modelName: String? = null) {
        currentDownloadCount = downloadCount
        currentModelName = modelName
        Log.d(TAG, "updateNotification count=$downloadCount modelName=$modelName")
        enterForeground()
    }

    override fun onDestroy() {
        super.onDestroy()
        instance = null
        Log.d(TAG, "onDestroy")
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            stopForeground(STOP_FOREGROUND_REMOVE)
        } else {
            @Suppress("DEPRECATION")
            stopForeground(true)
        }
    }

    override fun onBind(intent: Intent): IBinder? {
        return null
    }

    companion object {
        private const val TAG = "DownloadForegroundSvc"
        private const val CHANNEL_ID = "DownloadServiceChannel"
        private const val SERVICE_ID = 8888
        const val EXTRA_DOWNLOAD_COUNT = "download_count"
        const val EXTRA_MODEL_NAME = "model_name"
        
        @Volatile
        private var instance: DownloadForegroundService? = null
        
        fun getInstance(): DownloadForegroundService? = instance
        
        fun updateNotification(downloadCount: Int, modelName: String? = null) {
            instance?.updateNotification(downloadCount, modelName)
        }
    }
}
