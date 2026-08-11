package com.alibaba.mnnllm.api.openai.network.application

import com.alibaba.mnnllm.api.openai.network.routes.anthropicRoutes
import com.alibaba.mnnllm.api.openai.network.routes.chatRoutes
import com.alibaba.mnnllm.api.openai.network.routes.modelsRoutes
import com.alibaba.mnnllm.api.openai.network.routes.queueRoutes
import io.ktor.http.ContentType
import io.ktor.server.application.Application
import io.ktor.server.application.call
import io.ktor.server.application.install
import io.ktor.server.auth.authenticate
import io.ktor.server.plugins.calllogging.CallLogging
import io.ktor.server.request.httpMethod
import io.ktor.server.request.path
import io.ktor.server.request.uri
import io.ktor.server.response.respondText
import io.ktor.server.routing.get
import io.ktor.server.routing.routing
import io.ktor.server.sse.SSE
import io.ktor.server.sse.sse
import io.ktor.sse.ServerSentEvent
import org.slf4j.event.Level
import java.io.InputStream

fun Application.configureRouting() {
    install(SSE)

    routing {
        get("/") {
            try {
                val htmlContent = loadHtmlFromAssets()
                call.respondText(htmlContent, contentType = ContentType.Text.Html)
            } catch (e: Exception) {
                call.respondText("Error loading test page: ${e.message}", contentType = ContentType.Text.Plain)
            }
        }

        sse("/hello") {
            send(ServerSentEvent("world"))
        }

        queueRoutes()
        modelsRoutes()

        // Anthropic-compatible endpoint (/v1/messages) with x-api-key or bearer auth.
        anthropicRoutes()

        authenticate("auth-bearer") {
            chatRoutes()
        }
    }

    install(CallLogging) {
        level = Level.INFO
        filter { call ->
            val path = call.request.path()
            path.startsWith("/v1/chat/completions") || path.startsWith("/v1/messages")
        }
        format { call ->
            val status = call.response.status()
            val method = call.request.httpMethod.value
            val uri = call.request.uri
            val userAgent = call.request.headers["User-Agent"] ?: "Unknown"
            "$status: $method $uri - User-Agent: $userAgent"
        }
    }
}

private fun loadHtmlFromAssets(): String {
    return try {
        val context = com.alibaba.mnnllm.android.MnnLlmApplication.getAppContext()
        val inputStream: InputStream = context.assets.open("test_page.html")
        inputStream.bufferedReader().use { reader ->
            reader.readText()
        }
    } catch (e: Exception) {
        throw Exception("Failed to load HTML from assets: ${e.message}")
    }
}
