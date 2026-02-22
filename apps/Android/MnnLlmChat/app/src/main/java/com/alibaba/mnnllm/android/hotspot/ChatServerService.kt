package com.alibaba.mnnllm.android.hotspot

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.os.Binder
import android.os.Build
import android.os.IBinder
import androidx.core.app.NotificationCompat
import androidx.core.app.ServiceCompat
import android.content.pm.ServiceInfo
import com.alibaba.mnnllm.android.R

class ChatServerService : Service() {

    private var serverManager: ChatServerManager? = null

    inner class LocalBinder : Binder() {
        fun getManager(): ChatServerManager? = serverManager
    }

    private val binder = LocalBinder()

    override fun onBind(intent: Intent): IBinder = binder

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START -> {
                val modelId = intent.getStringExtra(EXTRA_MODEL_ID) ?: return START_NOT_STICKY
                val configPath = intent.getStringExtra(EXTRA_CONFIG_PATH) ?: return START_NOT_STICKY
                startForegroundWithNotification()
                serverManager = ChatServerManager.create(this)
                serverManager!!.start(modelId, configPath)
            }
            ACTION_STOP -> stopSelf()
        }
        return START_NOT_STICKY
    }

    private fun startForegroundWithNotification() {
        val channelId = "chat_server_channel"
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                channelId, "Chat Server", NotificationManager.IMPORTANCE_LOW
            )
            getSystemService(NotificationManager::class.java).createNotificationChannel(channel)
        }
        val notification = NotificationCompat.Builder(this, channelId)
            .setContentTitle(getString(R.string.chat_server_running))
            .setContentText(getString(R.string.chat_server_notification_text, CHAT_SERVER_PORT))
            .setSmallIcon(R.mipmap.ic_launcher)
            .setOngoing(true)
            .build()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            ServiceCompat.startForeground(this, NOTIF_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
        } else {
            startForeground(NOTIF_ID, notification)
        }
    }

    override fun onDestroy() {
        serverManager?.stop()
        serverManager = null
        super.onDestroy()
    }

    companion object {
        const val ACTION_START = "com.alibaba.mnnllm.android.hotspot.ACTION_START"
        const val ACTION_STOP  = "com.alibaba.mnnllm.android.hotspot.ACTION_STOP"
        const val EXTRA_MODEL_ID    = "extra_model_id"
        const val EXTRA_CONFIG_PATH = "extra_config_path"
        private const val NOTIF_ID = 1001
    }
}