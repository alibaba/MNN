package com.alibaba.mnnllm.api.openai.manager

import io.ktor.server.application.Application
import io.ktor.server.application.ApplicationStarted
import io.ktor.server.application.ApplicationStarting
import io.ktor.server.application.ApplicationStopped
import io.ktor.server.application.ApplicationStopping
import io.ktor.server.application.ServerReady
import timber.log.Timber

class ServerEventMonitor(private val application: Application) {
    private val serverEventManager = ServerEventManager.getInstance()

    fun startMonitoring() {
        application.monitor.subscribe(ApplicationStarting) {
            Timber.Forest.tag("ServerEvent").i("Application starting...")
            serverEventManager.handleApplicationStarting()
        }

        application.monitor.subscribe(ApplicationStarted) {
            Timber.Forest.tag("ServerEvent").i("Application started")
            serverEventManager.handleApplicationStarted()
        }

        application.monitor.subscribe(ServerReady) {
            Timber.Forest.tag("ServerEvent").i("Server ready to accept connections")
            serverEventManager.handleServerReady()
        }

        application.monitor.subscribe(ApplicationStopping) {
            Timber.Forest.tag("ServerEvent").i("Application stopping...")
            serverEventManager.handleApplicationStopping()
        }

        application.monitor.subscribe(ApplicationStopped) {
            Timber.Forest.tag("ServerEvent").i("Application stopped")
            serverEventManager.handleApplicationStopped()
        }
    }
}