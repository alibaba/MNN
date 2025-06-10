package com.alibaba.mnnllm.api.openai.network.application


import android.content.Context
import io.ktor.server.application.Application

import io.ktor.server.engine.EmbeddedServer
import io.ktor.server.engine.embeddedServer
import io.ktor.server.netty.Netty

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

import timber.log.Timber
import java.net.InetSocketAddress
import java.net.Socket
import com.alibaba.mnnllm.api.openai.service.ApiServerConfig
import com.alibaba.mnnllm.api.openai.manager.ServerEventManager
import com.alibaba.mnnllm.api.openai.manager.ServerEventMonitor
import com.alibaba.mnnllm.api.openai.network.logging.LogCollector
import io.ktor.server.application.ApplicationStarted


class OpenAIApplication(private val lifecycleScope: CoroutineScope, private val context: Context) {
    private val mutex = Mutex()
    private var server: EmbeddedServer<*, *>? = null
    private val serverEventManager = ServerEventManager.getInstance()
    
    init {
        // 初始化配置
        ApiServerConfig.initializeConfig(context)
        // 初始化日志收集器
        LogCollector.getInstance().initialize()
    }

    fun start() {

        if (server != null) {
            Timber.tag("Network").w("Server is already running")
            return
        }

        try {
            // 从配置中获取端口和IP地址
            val port = ApiServerConfig.getPort(context)
            val ipAddress = ApiServerConfig.getIpAddress(context)
            
            Timber.tag("Network").i("Starting Ktor Server on $ipAddress:$port")
            Timber.tag("Network").d("Checking port availability...")
            
            Timber.tag("Network").i("Starting Ktor Server on $ipAddress:$port")
            checkPortAvailability(port)

            server = lifecycleScope.embeddedServer(
                Netty,
                port = port,
                host = ipAddress,
                module = { module(context, ipAddress, port) }
            ).start(wait = false) // 不阻塞当前协程
            
            Timber.tag("Network").i("Server started successfully on $ipAddress:$port")

        } catch (e: Exception) {
            Timber.e(e, "Failed to start Ktor server")
            stop()
        }
    }


    /**
     * 停止服务器
     * 
     * 异步停止服务器并处理可能的取消情况
     */
    fun stop() {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                mutex.withLock { 
                    Timber.tag("Network").d("Acquired lock for server stop")
                    stopInternal() 
                }
            } catch (e: Exception) {
                Timber.tag("Network").e(e, "Failed to stop server due to lock acquisition failure")
            }
        }.invokeOnCompletion { cause ->
            cause?.let {
                Timber.tag("Network").w("Server stop job completed with exception: ${it.message}")
            }
        }
    }


    fun isRunning(): Boolean = serverEventManager.isServerRunning().also {
        Timber.tag("Network").i("Server is running: $it")
    }
    
    fun isReady(): Boolean = serverEventManager.isServerReady()
    
    fun getServerState() = serverEventManager.getCurrentState()
    
    fun getServerInfo() = serverEventManager.getCurrentInfo()

    fun getPort(): Int = ApiServerConfig.getPort(context)

    private fun checkPortAvailability(port: Int): Boolean {
        return try {
            Socket().use { socket ->
                socket.connect(InetSocketAddress("localhost", port), 1000)
                Timber.tag("Network").e("Port $port 被使用")
                false // 已被占用

            }
        } catch (e: Exception) {
            Timber.tag("Network").i("Port $port 可用")
            true // 可用
        }
    }


    /**
     * 内部方法，用于安全停止服务器
     * 
     * 处理服务器停止过程中的各种异常情况，包括协程取消
     */
    suspend fun stopInternal() {
        try {
            Timber.tag("Network").d("Attempting to stop server...")
            server?.stopSuspend(gracePeriodMillis = 1000, timeoutMillis = 5000)?.let {
                Timber.tag("Network").i("Server stopped gracefully")
            }
        } catch (e: kotlinx.coroutines.CancellationException) {
            Timber.tag("Network").w("Server stop was cancelled: ${e.message}")
            // 即使被取消也尝试强制停止
            server?.stop(gracePeriodMillis = 0, timeoutMillis = 1000)
        } catch (e: Exception) {
            Timber.tag("Network").e(e, "Error stopping server")
        } finally {
            server = null
            Timber.tag("Network").i("Server stopped completely")
        }
    }
}


private fun Application.module(context: Context, host: String, port: Int) {
    // 配置服务器事件监听
    ServerEventMonitor(this).startMonitoring()

    //来自com/alibaba/mnnllm/api/openai/network/application/HTTP.kt
    configureHTTP(context)

    //来自com/alibaba/mnnllm/api/openai/network/application/Routing.kt
    configureRouting()
}
