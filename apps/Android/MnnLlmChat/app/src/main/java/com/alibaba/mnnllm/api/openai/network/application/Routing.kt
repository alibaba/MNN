package com.alibaba.mnnllm.api.openai.network.application


import com.alibaba.mnnllm.android.chat.ChatActivity
import com.alibaba.mnnllm.android.llm.LlmSession
import com.alibaba.mnnllm.api.openai.network.routes.chatRoutes
import io.ktor.http.*
import io.ktor.serialization.kotlinx.json.*
import io.ktor.server.application.*
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


fun Application.configureRouting() {
    install(SSE)

    routing {
        get("/") {
            val response = "Hello, World!"
            call.respondText(response, contentType = ContentType.Text.Plain)
        }
        sse("/hello") {
            send(ServerSentEvent("world"))
         }
        chatRoutes()

    }
}


