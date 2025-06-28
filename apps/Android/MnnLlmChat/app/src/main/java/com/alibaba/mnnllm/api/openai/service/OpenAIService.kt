package com.alibaba.mnnllm.api.openai.service

import android.app.Service
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.content.pm.ServiceInfo
import android.os.Binder
import android.os.Build
import android.os.IBinder
import androidx.annotation.RequiresApi
import com.alibaba.mnnllm.android.chat.ChatActivity
import com.alibaba.mnnllm.api.openai.service.ApiServiceCoordinator
import com.alibaba.mnnllm.api.openai.manager.ApiNotificationManager
import timber.log.Timber

class OpenAIService : Service() {
    private val TAG = this::class.java.simpleName
    private lateinit var coordinator: ApiServiceCoordinator

    companion object {
        private var isServiceRunning = false
        private var serviceConnection: ServiceConnection? = null

        fun startService(context: Context) {
            if (context !is ChatActivity) {
                Timber.tag("ServiceStartCondition").w("Invalid context. Not starting service.")
                return
            }

            val serviceIntent = Intent(context, OpenAIService::class.java)
            // 在启动服务前设置标志，避免onStartCommand中的检查失败
            isServiceRunning = true
            context.startForegroundService(serviceIntent)

            val connection = object : ServiceConnection {
                override fun onServiceConnected(name: ComponentName?, binder: IBinder?) {
                    val localBinder = binder as LocalBinder
                    localBinder.initialize()
                    serviceConnection = this
                }

                override fun onServiceDisconnected(name: ComponentName?) {
                    serviceConnection = null
                    // 服务断开连接时重置标志
                    isServiceRunning = false
                }
            }

            context.bindService(serviceIntent, connection, Context.BIND_AUTO_CREATE)
            serviceConnection = connection
        }

        /**
         * 释放服务资源并停止服务
         *
         * @param context 上下文对象
         * @param force 是否强制停止，默认为false
         */
        fun releaseService(context: Context, force: Boolean = false) {
            val serviceIntent = Intent(context, OpenAIService::class.java)
            
            try {
                if (force) {
                    context.stopService(serviceIntent)
                    Timber.tag("ServiceRelease").w("Service stopped forcefully")
                } else {
                    if (context.stopService(serviceIntent)) {
                        Timber.tag("ServiceRelease").i("Service stopped gracefully")
                    } else {
                        Timber.tag("ServiceRelease").w("Service was not running")
                    }
                }
            } catch (e: Exception) {
                Timber.tag("ServiceRelease").e(e, "Failed to stop service")
                if (force) {
                    try {
                        context.stopService(serviceIntent)
                        Timber.tag("ServiceRelease").w("Retry force stop succeeded")
                    } catch (e: Exception) {
                        Timber.tag("ServiceRelease").e(e, "Force stop also failed")
                    }
                }
            }

            serviceConnection?.let { conn ->
                try {
                    context.unbindService(conn)
                    Timber.tag("ServiceRelease").i("Unbound successfully")
                } catch (e: Exception) {
                    Timber.tag("ServiceRelease").w(e, "Failed to unbind service")
                    if (force) {
                        try {
                            context.unbindService(conn)
                            Timber.tag("ServiceRelease").w("Force unbind succeeded")
                        } catch (e: Exception) {
                            Timber.tag("ServiceRelease").e(e, "Force unbind also failed")
                        }
                    }
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
        if (!isServiceRunning) {
            Timber.tag("ServiceLifecycle").w("Service started illegally and will be stopped immediately.")
            stopSelf()
            return START_NOT_STICKY
        }
        val notification = coordinator.getNotification()
        if (notification != null) {
            startForeground(ApiNotificationManager.NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
        }
        return START_NOT_STICKY
    }




    override fun onCreate() {
        super.onCreate()
        coordinator = ApiServiceCoordinator(this)
        coordinator.initialize()
        
        val notification = coordinator.getNotification()
        if (notification != null) {
            startForeground(ApiNotificationManager.NOTIFICATION_ID, notification)
        }
    }

    override fun onDestroy() {
        Timber.tag(TAG).i("Service is being destroyed")
        cleanup()
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder {
        Timber.tag(TAG).d("Service bound by client")
        return LocalBinder()
    }

    inner class LocalBinder : Binder() {
        fun getService(): OpenAIService = this@OpenAIService
        fun initialize() = this@OpenAIService.initializeWithSession()
    }


    /**
     * 清理服务资源
     *
     * 包括停止前台服务和清理协调器资源
     */
    private fun cleanup() {
        try {
            coordinator.cleanup()
            Timber.tag(TAG).d("Coordinator cleanup completed")
        } catch (e: Exception) {
            Timber.tag(TAG).e(e, "Failed to cleanup coordinator")
        }

        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                stopForeground(STOP_FOREGROUND_REMOVE)
                Timber.tag(TAG).d("Foreground service stopped (API >= TIRAMISU)")
            } else {
                @Suppress("DEPRECATION")
                stopForeground(true)
                Timber.tag(TAG).d("Foreground service stopped (legacy API)")
            }
        } catch (e: Exception) {
            Timber.tag(TAG).e(e, "Failed to stop foreground service")
        } finally {
            Timber.tag(TAG).i("Service cleanup completed")
        }
    }


    fun initializeWithSession() {
        val success = coordinator.startServer()
        if (!success) {
            Timber.tag(TAG).w("Failed to start server through coordinator")
        }
    }

    fun updateNotification(contentTitle: String, contentText: String) {
        coordinator.updateNotification(contentTitle, contentText)
    }

    fun getServerPort(): Int? = coordinator.getServerPort()
    
    fun isServerRunning(): Boolean = coordinator.isServerRunning
}
