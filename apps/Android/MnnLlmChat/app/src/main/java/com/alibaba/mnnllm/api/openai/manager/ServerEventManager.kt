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

/** * serviceeventmanager * useKtoreventsystemto manageservicestate,providereactivestateupdate*/
class ServerEventManager {

    private val _serverState = MutableStateFlow(ServerState.STOPPED)
    val serverState: StateFlow<ServerState> = _serverState.asStateFlow()

    private val _serverInfo = MutableStateFlow(ServerInfo())
    val serverInfo: StateFlow<ServerInfo> = _serverInfo.asStateFlow()

    /** * servicestateenum*/
    enum class ServerState {
        STARTING,
        STARTED,
        READY,
        STOP_PREPARING,
        STOPPING,
        STOPPED
    }

    /** * serviceinfodataclass*/
    data class ServerInfo(
        val host: String = "",
        val port: Int = 0,
        val isRunning: Boolean = false,
        val startTime: Long = 0L
    )

    /** * checkservicewhethercurrentlyrunning*/
    fun isServerRunning(): Boolean {
        return _serverState.value in listOf(ServerState.STARTED, ServerState.READY)
    }

    /** * checkservicewhetheralreadyready*/
    fun isServerReady(): Boolean {
        return _serverState.value == ServerState.READY
    }

    /** * getcurrentservicestate*/
    fun getCurrentState(): ServerState {
        return _serverState.value
    }

    /** * getcurrentserviceinfo*/
    fun getCurrentInfo(): ServerInfo {
        return _serverInfo.value
    }

    /**
     * process ApplicationStarting event*/
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
     * process ApplicationStarted event*/
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
     * process ServerReady event*/
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
     * process ApplicationStopping event*/
    fun handleApplicationStopping() {
        Timber.Forest.tag("ServerEvent").i("Application stopping...")
        _serverState.value = ServerState.STOPPING
    }

    /**
     * process ApplicationStopped event*/
    fun handleApplicationStopped() {
        _serverState.value = ServerState.STOPPED
        val currentInfo = _serverInfo.value
        _serverInfo.value = currentInfo.copy(
            isRunning = false,
            startTime = 0L
        )
        Timber.Forest.tag("ServerEvent").i("Application stopped")
    }

    /** * resetservicestate (forrestartservicewhen) * onlyresetrunningstateandtime,preservehostandportconfiginfo*/
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