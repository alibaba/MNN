package com.alibaba.mnnllm.api.openai.network.routes

import OpenAIChatRequest
import com.alibaba.mnnllm.api.openai.network.logging.ChatLogger
import com.alibaba.mnnllm.api.openai.network.services.MNNChatService
import io.ktor.server.request.receive
import io.ktor.server.routing.Route
import io.ktor.server.routing.post
import java.util.UUID

/** * chatroutedefine * responsible fordefineAPIrouteandrequestdispatch，followsingleresponsibilityprinciple * allbusinesslogicalreadysplittodedicatedserviceclassin*/

/** * registerchatrelatedroute*/
fun Route.chatRoutes() {
    val MNNChatService = MNNChatService()
    val logger = ChatLogger()

    post("/v1/chat/completions") {
        val traceId = UUID.randomUUID().toString()
        logger.logRequestStart(traceId, call)
        val chatRequest = call.receive<OpenAIChatRequest>()
        MNNChatService.processChatCompletion(call, chatRequest, traceId)
    }
}