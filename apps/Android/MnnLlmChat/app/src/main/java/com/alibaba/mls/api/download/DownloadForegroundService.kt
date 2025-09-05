package com.alibaba.mls.api.download

import android.R
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.content.pm.ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC
import android.os.Build
import android.os.IBinder
import com.alibaba.mnnllm.android.R as AppR
import com.alibaba.mnnllm.android.main.MainActivity

class DownloadForegroundService : Service() {
    private lateinit var notificationManager: NotificationManager
    private var currentDownloadCount = 0
    private var currentModelName: String? = null

    override fun onCreate() {
        super.onCreate()
        notificationManager = getSystemService(NotificationManager::class.java)
        createNotificationChannel()
        instance = this
    }

    private fun createNotificationChannel() {
        val channel = NotificationChannel(
            CHANNEL_ID, 
            getString(AppR.string.download_service_title), 
            NotificationManager.IMPORTANCE_DEFAULT
        )
        channel.description = "Shows download progress for model files"
        notificationManager.createNotificationChannel(channel)
    }

    override fun onStartCommand(intent: Intent, flags: Int, startId: Int): Int {
        // Extract download count and model name from intent if available
        currentDownloadCount = intent.getIntExtra(EXTRA_DOWNLOAD_COUNT, 1)
        currentModelName = intent.getStringExtra(EXTRA_MODEL_NAME)
        
        val notification = createNotification()
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                startForeground(SERVICE_ID, notification, FOREGROUND_SERVICE_TYPE_DATA_SYNC)
            } else {
                startForeground(SERVICE_ID, notification)
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
        return START_NOT_STICKY
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

        return Notification.Builder(this, CHANNEL_ID)
            .setContentTitle(contentTitle)
            .setContentText(contentText)
            .setSmallIcon(R.drawable.stat_sys_download)
            .setContentIntent(pendingIntent)
            .setAutoCancel(false)
            .setOngoing(true)
            .build()
    }

    fun updateNotification(downloadCount: Int, modelName: String? = null) {
        currentDownloadCount = downloadCount
        currentModelName = modelName
        android.util.Log.d("DownloadForegroundService", "updateNotification: count=$downloadCount, modelName=$modelName")
        val notification = createNotification()
        notificationManager.notify(SERVICE_ID, notification)
        android.util.Log.d("DownloadForegroundService", "Notification updated successfully")
    }

    override fun onDestroy() {
        super.onDestroy()
        instance = null
    }

    override fun onBind(intent: Intent): IBinder? {
        return null
    }

    companion object {
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
