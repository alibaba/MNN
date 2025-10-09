package com.alibaba.mnnllm.api.openai.network.application


import com.alibaba.mnnllm.android.chat.ChatActivity
import com.alibaba.mnnllm.android.llm.LlmSession
import com.alibaba.mnnllm.api.openai.network.routes.chatRoutes
import com.alibaba.mnnllm.api.openai.network.routes.queueRoutes
import com.alibaba.mnnllm.api.openai.network.routes.modelsRoutes
import io.ktor.http.*
import io.ktor.serialization.kotlinx.json.*
import io.ktor.server.application.*
import io.ktor.server.auth.authenticate
import io.ktor.server.plugins.calllogging.*
import io.ktor.server.plugins.contentnegotiation.*
import io.ktor.server.plugins.cors.routing.*
import io.ktor.server.plugins.statuspages.*
import io.ktor.server.request.*
import io.ktor.server.response.*
import io.ktor.server.routing.*
import io.ktor.server.sse.*
import io.ktor.sse.*
import org.slf4j.event.*
import android.content.Context
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
        
        // 队列状态路由 - 不需要认证
        queueRoutes()

        authenticate("auth-bearer") {
            // 在这里定义需要认证的路由
            // /v1/chat/completions
            chatRoutes()
            // /v1/models
            modelsRoutes()
        }
    }

    install(CallLogging) {
        level = Level.INFO
        filter { call ->
            call.request.path().startsWith("/v1/chat/completions")
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

/**
 * 从assets目录加载HTML文件
 */
private fun loadHtmlFromAssets(): String {
    return try {
        // 使用Application Context来访问assets
        val context = com.alibaba.mnnllm.android.MnnLlmApplication.getAppContext()
        val inputStream: InputStream = context.assets.open("test_page.html")
        
        inputStream.bufferedReader().use { reader ->
            reader.readText()
        }
    } catch (e: Exception) {
        throw Exception("Failed to load HTML from assets: ${e.message}")
    }
}