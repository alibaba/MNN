package com.alibaba.mnnllm.api.openai.network.routes

import com.alibaba.mnnllm.android.MnnLlmApplication
import com.alibaba.mnnllm.api.openai.network.auth.ApiAuthUtils
import com.alibaba.mnnllm.api.openai.network.compat.AnthropicMessagesRequest
import com.alibaba.mnnllm.api.openai.network.logging.ChatLogger
import com.alibaba.mnnllm.api.openai.network.services.AnthropicMessagesService
import io.ktor.server.request.receive
import io.ktor.server.routing.Route
import io.ktor.server.routing.post
import java.util.UUID

fun Route.anthropicRoutes() {
    val service = AnthropicMessagesService()
    val logger = ChatLogger()

    post("/v1/messages") {
        val context = MnnLlmApplication.getAppContext()
        if (!ApiAuthUtils.isAuthorized(call, context)) {
            ApiAuthUtils.respondUnauthorized(call)
            return@post
        }

        val traceId = UUID.randomUUID().toString()
        logger.logRequestStart(traceId, call)
        val request = call.receive<AnthropicMessagesRequest>()
        service.processMessages(call, request, traceId)
    }
}
