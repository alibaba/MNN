package com.alibaba.mnnllm.api.openai.service

import android.content.Context
import com.alibaba.mnnllm.api.openai.di.ServiceLocator
import com.alibaba.mnnllm.api.openai.manager.ApiNotificationManager
import com.alibaba.mnnllm.api.openai.network.application.OpenAIApplication
import com.alibaba.mnnllm.android.R
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.cancel
import timber.log.Timber

/**
 * 统一调度管理器，负责协调通知栏服务和服务器的生命周期
 */
class ApiServiceCoordinator(private val context: Context) {
    private val TAG = this::class.java.simpleName
    private val networkServiceScope = CoroutineScope(Dispatchers.IO)
    private var notificationManager: ApiNotificationManager? = null
    private var application: OpenAIApplication? = null

    private var _isInitialized = false
    val isInitialized: Boolean get() = _isInitialized

    private var _isServerRunning = false
    val isServerRunning: Boolean get() = _isServerRunning

    /**
     * 初始化协调器
     */
    fun initialize(): Boolean {
        return runCatching {
            if (_isInitialized) {
                Timber.Forest.tag(TAG).w("Coordinator already initialized")
                return true
            }

            // 初始化通知管理器
            notificationManager = ApiNotificationManager(context)

            _isInitialized = true
            Timber.Forest.tag(TAG).i("Coordinator initialized successfully")
            true
        }.getOrElse { e ->
            Timber.Forest.tag(TAG).e(e, "Failed to initialize coordinator")
            false
        }
    }

    /**
     * 启动服务器和通知
     */
    fun startServer(): Boolean {
        if (!_isInitialized) {
            Timber.Forest.tag(TAG).w("Coordinator not initialized")
            return false
        }

        return runCatching {
            val session = ServiceLocator.getChatSessionProvider().getLlmSession()
            if (session == null) {
                Timber.Forest.tag(TAG).w("No active LlmSession found")
                notificationManager?.updateNotification(
                    context.getString(R.string.api_service_not_started),
                    context.getString(R.string.no_active_session)
                )
                return false
            }

            // 确保ServerEventManager状态正确初始化
            // 如果之前被重置过，需要重新准备状态
            val serverEventManager = com.alibaba.mnnllm.api.openai.manager.ServerEventManager.getInstance()
            if (serverEventManager.getCurrentState() == com.alibaba.mnnllm.api.openai.manager.ServerEventManager.ServerState.STOPPED) {
                Timber.Forest.tag(TAG).d("ServerEventManager is in STOPPED state, ready for new server")
            }

            // 启动服务器
            val app = OpenAIApplication(networkServiceScope, context)
            app.start()
            application = app

            // 更新通知
            notificationManager?.updateNotification(
                context.getString(R.string.api_service_running),
                context.getString(R.string.api_service_port, app.getPort())
            )

            _isServerRunning = true
            Timber.Forest.tag(TAG).i("Server started successfully on port ${app.getPort()}")
            true
        }.getOrElse { e ->
            Timber.Forest.tag(TAG).e(e, "Failed to start server: ${e.message}")
            notificationManager?.updateNotification(
                context.getString(R.string.api_service_start_failed),
                context.getString(R.string.api_service_error, e.message)
            )
            false
        }
    }

    /**
     * 停止服务器和通知
     */
    fun stopServer() {
        runCatching {
            // 停止服务器
            application?.stop()
            application = null

            com.alibaba.mnnllm.api.openai.manager.ServerEventManager.getInstance().resetRuntimeState()
            Timber.Forest.tag(TAG).d("ServerEventManager state reset after application is nullified.")

            // 取消通知
            notificationManager?.cancelNotification()

            _isServerRunning = false
            Timber.Forest.tag(TAG).i("Server stopped successfully")
        }.onFailure { e ->
            Timber.Forest.tag(TAG).e(e, "Error stopping server: ${e.message}")
        }
    }

    /**
     * 更新通知内容
     */
    fun updateNotification(title: String, content: String) {
        notificationManager?.updateNotification(title, content)
    }

    /**
     * 获取通知对象（用于前台服务）
     */
    fun getNotification(
        title: String = context.getString(R.string.api_service_running),
        content: String = context.getString(R.string.api_service_port, 8080)
    ) = notificationManager?.buildNotification(title, content)

    /**
     * 获取服务器端口
     */
    fun getServerPort(): Int? = application?.getPort()

    /**
     * 检查服务器是否运行
     */
    fun checkServerStatus(): Boolean = application?.isRunning() ?: false

    /**
     * 检查服务器是否就绪
     */
    fun checkServerReady(): Boolean = application?.isReady() ?: false

    /**
     * 获取服务器状态
     */
    fun getServerState() = application?.getServerState()

    /**
     * 获取服务器信息
     */
    fun getServerInfo() = application?.getServerInfo()

    /**
     * 清理资源
     */
    fun cleanup() {
        runCatching {
            val appToStop = application
            if (appToStop != null) {
                Timber.Forest.tag(TAG).i("Cleanup: Requesting server stop for application: $appToStop")
                appToStop.stop() // 发出停止请求
                
                Timber.Forest.tag(TAG).i("Cleanup: Waiting 5 seconds for server to stop gracefully...")
                try {
                    Thread.sleep(3000) // 增加等待时间到5秒
                } catch (e: InterruptedException) {
                    Thread.currentThread().interrupt()
                    Timber.Forest.tag(TAG).w("Cleanup delay interrupted after server stop request.")
                }
                Timber.Forest.tag(TAG).i("Cleanup: Finished waiting. Proceeding with coordinator cleanup.")
                application = null // 在等待之后置空
            }

            // 确保即使 appToStop 为 null，也尝试重置状态和取消通知
            com.alibaba.mnnllm.api.openai.manager.ServerEventManager.getInstance().resetRuntimeState()
            Timber.Forest.tag(TAG).d("ServerEventManager state reset during cleanup.")
            notificationManager?.cancelNotification()
            
            Timber.Forest.tag(TAG).i("Cleanup: Cancelling networkServiceScope.")
            networkServiceScope.cancel() // 最后取消作用域
            notificationManager = null
            _isInitialized = false
            Timber.Forest.tag(TAG).i("Coordinator cleaned up")
        }.onFailure { e ->
            Timber.Forest.tag(TAG).e(e, "Error during cleanup: ${e.message}")
        }
    }
}