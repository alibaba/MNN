package com.alibaba.mnnllm.api.openai.service

import android.app.Notification
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.IBinder
import androidx.core.app.NotificationCompat
import timber.log.Timber

import com.alibaba.mnnllm.android.llm.LlmSession
import com.alibaba.mnnllm.api.openai.network.application.OpenAIApplication
import android.app.NotificationChannel
import android.content.ComponentName
import android.content.ServiceConnection
import android.content.pm.ServiceInfo
import android.os.Binder
import androidx.annotation.RequiresApi
import com.alibaba.mnnllm.android.chat.ChatActivity
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class OpenAIService : Service() {
    private val TAG = this::class.java.simpleName
    private val CHANNEL_ID = "api_service_channel"

    // 使用普通引用替代 WeakReference
    private var application: OpenAIApplication? = null
    private val networkServiceScope = CoroutineScope(Dispatchers.IO)
    private lateinit var notificationManager: NotificationManager





    companion object {
        const val NOTIFICATION_ID = 1001
        private const val DEFAULT_PORT_MESSAGE = "正在监听端口：8080"
        private const val LOG_TAG_BASE = "Step Base"
        private const val NOTIFICATION_TITLE = "MNN API 服务运行中"
        private const val NOTIFICATION_CHANNEL_NAME = "API 服务通道"
        private const val NOTIFICATION_CHANNEL_DESCRIPTION = "用于显示 API 服务的持久通知"

        // 保存当前绑定的服务连接对象
        private var isServiceRunning = false
        private var serviceConnection: ServiceConnection? = null

        fun startWithSession(context: Context, session: LlmSession) {
            // 添加启动条件判断，确保Context为ChatActivity且LlmSession有效
            if (context !is ChatActivity || session == null) {
                Timber.tag("ServiceStartCondition").w("Invalid context or session. Not starting service.")
                return
            }

            val serviceIntent = Intent(context, OpenAIService::class.java)
            context.startForegroundService(serviceIntent)

            val connection = object : ServiceConnection {
                override fun onServiceConnected(name: ComponentName?, binder: IBinder?) {
                    val localBinder = binder as LocalBinder
                    localBinder.initialize(session)
                    serviceConnection = this
                }

                override fun onServiceDisconnected(name: ComponentName?) {
                    serviceConnection = null
                }
            }

            context.bindService(serviceIntent, connection, Context.BIND_AUTO_CREATE)
            serviceConnection = connection
        }



        fun releaseService(context: Context) {
            val serviceIntent = Intent(context, OpenAIService::class.java)
            try {
                context.stopService(serviceIntent)
                Timber.tag("ServiceRelease").i("Service stopped")
            } catch (e: Exception) {
                Timber.tag("ServiceRelease").w(e, "Failed to stop service")
            }

            if (serviceConnection != null) {
                try {
                    context.unbindService(serviceConnection!!)
                    Timber.tag("ServiceRelease").i("Unbound successfully")
                } catch (e: Exception) {
                    Timber.tag("ServiceRelease").w(e, "Failed to unbind service")
                } finally {
                    serviceConnection = null
                }
            }

            isServiceRunning = false
            Timber.tag("ServiceLifecycle").i("OpenAIService resources released")
        }
    }



@RequiresApi(Build.VERSION_CODES.Q)
override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
    // 防止因系统重启或自启动权限导致的非法唤起
    if (!isServiceRunning) {
        Timber.tag("ServiceLifecycle").w("Service started illegally and will be stopped immediately.")
        stopSelf()
        return START_NOT_STICKY
    }
    val notification = buildNotification()
    startForeground(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)    // 不希望服务被系统自动重启
    return START_NOT_STICKY
}

    private fun buildNotification(contentTitle: String = "API 服务运行中", contentText: String = "正在监听端口：8080"): Notification {
        return NotificationCompat.Builder(this, CHANNEL_ID).apply {
            setContentTitle(contentTitle)
            setContentText(contentText)
            setSmallIcon(android.R.drawable.ic_dialog_info)
            setPriority(NotificationCompat.PRIORITY_HIGH)
            setOngoing(true)
            setAutoCancel(false)
            setVisibility(NotificationCompat.VISIBILITY_PUBLIC)
        }.build()
    }


    private fun initializeNotification() {
        val channel = NotificationChannel(CHANNEL_ID, NOTIFICATION_CHANNEL_NAME, NotificationManager.IMPORTANCE_HIGH)
        notificationManager.createNotificationChannel(channel)
    }


    override fun onCreate() {
        super.onCreate()
        notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        initializeNotification()

        // ✅ 立即调用
        val notification = buildNotification()
        startForeground(NOTIFICATION_ID, notification)
    }

    override fun onDestroy() {
        Timber.tag(TAG).i("Service is being destroyed")

        extracted()

        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder {
        Timber.tag(TAG).d("Service bound by client")
        return LocalBinder()
    }
    inner class LocalBinder : Binder() {
        fun getService(): OpenAIService = this@OpenAIService
        fun initialize(session: LlmSession) = this@OpenAIService.initializeWithSession(session)
    }


    private fun extracted() {
        networkServiceScope.cancel()
        application?.stop()
        application = null

        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                stopForeground(STOP_FOREGROUND_REMOVE)
            } else {
                @Suppress("DEPRECATION")
                stopForeground(true)
            }
        } catch (e: Exception) {
            Timber.tag(TAG).w("Failed to stop foreground service: ${e.message}")
        }

        try {
            notificationManager.cancel(NOTIFICATION_ID)
        } catch (e: Exception) {
            Timber.tag(TAG).w("Failed to cancel notification: ${e.message}")
        }
    }


    fun initializeWithSession(session: LlmSession) {

            runCatching {
                val app = OpenAIApplication(session, networkServiceScope)
                app.start()
                bindApplication(app)
            }.onFailure { e ->
                Timber.tag(TAG).e(e, "Initialization failed: ${e.message}")
            }

    }

    fun bindApplication(application: OpenAIApplication?) {
        this.application = application
        if (application != null) {
            updateNotification("API 服务运行中", "端口：${application.getPort()}")
            Timber.tag("ServiceStartup").i("Server started on port ${application.getPort()}")        } else {
            Timber.tag("ServiceLifecycle").w("Attempted to bind a null OpenAIApplication")
            updateNotification("API 服务未启动", "服务初始化失败")
        }
    }

    fun updateNotification(contentTitle: String, contentText: String) {
        val notificationBuilder = NotificationCompat.Builder(this, CHANNEL_ID).apply {
            setContentTitle(contentTitle)
            setContentText(contentText)
            setSmallIcon(android.R.drawable.ic_dialog_info)
            setPriority(NotificationCompat.PRIORITY_HIGH)
            setOngoing(true)
            setAutoCancel(false)
            setVisibility(NotificationCompat.VISIBILITY_PUBLIC)
        }

        Timber.tag("ForegroundService").i("Updating notification: $contentTitle - $contentText")
        notificationManager.notify(NOTIFICATION_ID, notificationBuilder.build())
    }


}
