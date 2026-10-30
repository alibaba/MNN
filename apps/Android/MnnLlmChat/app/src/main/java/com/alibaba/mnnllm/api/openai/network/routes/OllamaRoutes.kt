package com.alibaba.mnnllm.api.openai.network.routes

import com.alibaba.mnnllm.api.openai.di.ServiceLocator
import com.alibaba.mnnllm.api.openai.network.compat.OllamaAdapter
import com.alibaba.mnnllm.api.openai.network.services.OllamaService
import io.ktor.http.ContentType
import io.ktor.http.HttpStatusCode
import io.ktor.server.request.receive
import io.ktor.server.response.respondText
import io.ktor.server.routing.Route
import io.ktor.server.routing.get
import io.ktor.server.routing.post
import kotlinx.coroutines.CancellationException
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import java.util.UUID

fun Route.ollamaRoutes() {
    val service = OllamaService()
    get("/api/tags") {
        val model = ServiceLocator.getChatSessionProvider().getLlmSession()?.modelId()
        val result = buildJsonObject {
            put("models", buildJsonArray {
                if (model != null) add(buildJsonObject { put("name", model); put("model", model) })
            })
        }
        call.respondText(result.toString(), ContentType.Application.Json)
    }
    for ((path, chat) in listOf("/api/chat" to true, "/api/generate" to false)) {
        post(path) {
            val request = try {
                OllamaAdapter.parse(call.receive<JsonObject>(), chat)
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                val error = buildJsonObject { put("error", e.message ?: "Invalid request") }
                call.respondText(error.toString(), ContentType.Application.Json, HttpStatusCode.BadRequest)
                return@post
            }
            service.generate(call, request, chat, UUID.randomUUID().toString())
        }
    }
}
