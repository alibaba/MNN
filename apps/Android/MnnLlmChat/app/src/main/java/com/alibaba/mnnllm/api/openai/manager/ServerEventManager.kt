package com.alibaba.mnnllm.api.openai.manager

import io.ktor.server.application.Application
import io.ktor.server.application.ApplicationStarted
import io.ktor.server.application.ApplicationStarting
import io.ktor.server.application.ApplicationStopPreparing
import io.ktor.server.application.ApplicationStopped
import io.ktor.server.application.ApplicationStopping
import io.ktor.server.application.ServerReady
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import timber.log.Timber

/**
 * 服务器事件管理器
 * 使用Ktor的事件系统来管理服务器状态，提供响应式的状态更新
 */
class ServerEventManager {

    private val _serverState = MutableStateFlow(ServerState.STOPPED)
    val serverState: StateFlow<ServerState> = _serverState.asStateFlow()

    private val _serverInfo = MutableStateFlow(ServerInfo())
    val serverInfo: StateFlow<ServerInfo> = _serverInfo.asStateFlow()

    /**
     * 服务器状态枚举
     */
    enum class ServerState {
        STARTING,
        STARTED,
        READY,
        STOP_PREPARING,
        STOPPING,
        STOPPED
    }

    /**
     * 服务器信息数据类
     */
    data class ServerInfo(
        val host: String = "",
        val port: Int = 0,
        val isRunning: Boolean = false,
        val startTime: Long = 0L
    )

    /**
     * 检查服务器是否正在运行
     */
    fun isServerRunning(): Boolean {
        return _serverState.value in listOf(ServerState.STARTED, ServerState.READY)
    }

    /**
     * 检查服务器是否已就绪
     */
    fun isServerReady(): Boolean {
        return _serverState.value == ServerState.READY
    }

    /**
     * 获取当前服务器状态
     */
    fun getCurrentState(): ServerState {
        return _serverState.value
    }

    /**
     * 获取当前服务器信息
     */
    fun getCurrentInfo(): ServerInfo {
        return _serverInfo.value
    }

    /**
     * 处理 ApplicationStarting 事件
     */
    fun handleApplicationStarting(host: String = "", port: Int = 0) {
        Timber.Forest.tag("ServerEvent").i("Application starting...")
        _serverState.value = ServerState.STARTING
        _serverInfo.value = _serverInfo.value.copy(
            host = host,
            port = port,
            isRunning = false
        )
    }

    /**
     * 处理 ApplicationStarted 事件
     */
    fun handleApplicationStarted(host: String = "", port: Int = 0) {
        _serverState.value = ServerState.STARTED
        val currentInfo = _serverInfo.value
        _serverInfo.value = currentInfo.copy(
            isRunning = true,
            startTime = System.currentTimeMillis(),
            host = host.ifEmpty { currentInfo.host },
            port = if (port != 0) port else currentInfo.port
        )
        Timber.Forest.tag("ServerEvent").i("Application started")
    }

    /**
     * 处理 ServerReady 事件
     */
    fun handleServerReady(host: String = "", port: Int = 0) {
        _serverState.value = ServerState.READY
        val currentInfo = _serverInfo.value
        _serverInfo.value = currentInfo.copy(
            isRunning = true,
            host = host.ifEmpty { currentInfo.host },
            port = if (port != 0) port else currentInfo.port
        )
        Timber.Forest.tag("ServerEvent").i("Server ready to accept connections")
    }

    /**
     * 处理 ApplicationStopping 事件
     */
    fun handleApplicationStopping() {
        Timber.Forest.tag("ServerEvent").i("Application stopping...")
        _serverState.value = ServerState.STOPPING
    }

    /**
     * 处理 ApplicationStopped 事件
     */
    fun handleApplicationStopped() {
        _serverState.value = ServerState.STOPPED
        val currentInfo = _serverInfo.value
        _serverInfo.value = currentInfo.copy(
            isRunning = false,
            startTime = 0L
        )
        Timber.Forest.tag("ServerEvent").i("Application stopped")
    }

    /**
     * 重置服务器状态（用于重启服务时）
     * 只重置运行状态和时间，保留host和port配置信息
     */
    fun resetRuntimeState() {
        Timber.Forest.tag("ServerEvent").i("Resetting ServerEventManager runtime state for restart")
        _serverState.value = ServerState.STOPPED
        val currentInfo = _serverInfo.value
        _serverInfo.value = currentInfo.copy(
            isRunning = false,
            startTime = 0L
        )
        Timber.Forest.tag("ServerEvent").d("ServerEventManager runtime state reset complete, ready for new server configuration")
    }

    companion object {
        @Volatile
        private var INSTANCE: ServerEventManager? = null

        fun getInstance(): ServerEventManager {
            return INSTANCE ?: synchronized(this) {
                INSTANCE ?: ServerEventManager().also { INSTANCE = it }
            }
        }
    }
}