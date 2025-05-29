package com.alibaba.mnnllm.api.openai.network.application

import com.alibaba.mnnllm.android.llm.LlmSession
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpMethod
import io.ktor.serialization.kotlinx.json.json
import io.ktor.server.application.Application
import io.ktor.server.application.ApplicationCallPipeline
import io.ktor.server.application.install
import io.ktor.server.engine.EmbeddedServer
import io.ktor.server.engine.embeddedServer
import io.ktor.server.netty.Netty
import io.ktor.server.plugins.calllogging.CallLogging
import io.ktor.server.plugins.contentnegotiation.ContentNegotiation
import io.ktor.server.plugins.cors.routing.CORS
import io.ktor.server.plugins.doublereceive.DoubleReceive
import io.ktor.server.request.httpMethod
import io.ktor.server.request.receiveText
import io.ktor.server.request.uri
import io.ktor.util.flattenEntries
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import timber.log.Timber
import java.net.InetSocketAddress
import java.net.Socket
import kotlin.coroutines.coroutineContext


class OpenAIApplication(private val llmSession: LlmSession, private val lifecycleScope: CoroutineScope) {
    private val mutex = Mutex()
    private var _running = false
    var running: Boolean
        get() = _running
        private set(value) { _running = value }

    private val port = 8080
    private var server: EmbeddedServer<*, *>? = null
    private var serverJob: Job? = null

    fun start() {

            if (server != null) {
                Timber.tag("Network").w("Server is already running")

            }

            try {
                Timber.tag("Network").i("Starting Ktor Server on Port: $port")
                checkPortAvailability(port)

                lifecycleScope.embeddedServer(
                        Netty,
                        port = 8080,
                        host = "0.0.0.0",
                        module = Application::module
                ).start(wait = false) // 不阻塞当前协程

              //  running = confirmServerStarted(port)
                Timber.tag("Network").i("Server running: $running")

            } catch (e: Exception) {
                Timber.e(e, "Failed to start Ktor server")
                stop()
            }
    }





    fun stop() {
        lifecycleScope.launch(Dispatchers.IO) {
            mutex.withLock { stopInternal() }
        }
    }



    fun isRunning(): Boolean = running.also {
        Timber.tag("Network").i("Server is running: $running")
    }

    fun getPort(): Int = port

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

    private suspend fun confirmServerStarted(port: Int): Boolean {
        repeat(5) {
            try {
                Socket().use { it.connect(InetSocketAddress("127.0.0.1", port), 1000) }
                return true
            } catch (_: Exception) {
                delay(1000)
            }
        }
        return false
    }





    private suspend fun healthCheckLoop() {
        // 使用 coroutineContext.isActive 检查当前协程是否活跃
        while (coroutineContext.isActive) {
            delay(30_000)
            if (!confirmServerAlive()) {
                Timber.tag("ServerHealth").e("Server not responding")
                stopInternal()
                break
            }
        }
    }


    private fun confirmServerAlive(): Boolean = runCatching {
        Socket().use { it.connect(InetSocketAddress("127.0.0.1", port), 1000) }
        true
    }.getOrElse { false }

     suspend fun stopInternal() {


        try {
            server?.stopSuspend(gracePeriodMillis = 1000, timeoutMillis = 5000)
        } catch (e: Exception) {
            Timber.e(e, "Error stopping server")
        } finally {
                server = null
             running = false
            Timber.tag("Network").i("Server stopped completely")
        }
    }
}


private fun Application.module() {
/***
    configurePlugins()
    routing {
        get("/") {
            val response = "Hello, World!"
            call.respondText(response, contentType = ContentType.Text.Plain)
        }
        chatRoutes()
    }
***/
    //来自com/alibaba/mnnllm/api/openai/network/application/HTTP.kt
    configureHTTP()

    //来自com/alibaba/mnnllm/api/openai/network/application/Routing.kt
    configureRouting()

  }




private fun Application.configurePlugins() {
    install(DoubleReceive)
    install(ContentNegotiation) { json() }
    install(CORS) {
        allowHost("localhost")
        allowHost("0.0.0.0:8080")
        allowMethod(HttpMethod.Post)
        allowHeader(HttpHeaders.Authorization)
        allowHeader(HttpHeaders.ContentType)
        allowCredentials = true
    }
    install(CallLogging) {
        format { call ->
            "Status: ${call.response.status()}, Method: ${call.request.httpMethod.value}"
        }
    }
    intercept(ApplicationCallPipeline.Call) {
        Timber.tag("GlobalPreflight").d(
            """
                Request: ${context.request.httpMethod.value} ${context.request.uri}
                Headers: ${context.request.headers.flattenEntries().joinToString("; ")}
                """.trimIndent()
        )
        if (context.request.httpMethod in listOf(HttpMethod.Post, HttpMethod.Put)) {
            try {
                val body = context.receiveText()
                if (body.isNotBlank()) Timber.tag("GlobalPreflight").d("Request Body: $body")
            } catch (e: Exception) {
                Timber.tag("GlobalPreflight").w(e, "Failed to read request body.")
            }
        }
    }
}