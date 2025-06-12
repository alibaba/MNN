package com.alibaba.mnnllm.api.openai.network.routes

import OpenAIChatRequest
import com.alibaba.mnnllm.api.openai.network.logging.ChatLogger
import com.alibaba.mnnllm.api.openai.network.services.MNNChatService
import io.ktor.server.request.receive
import io.ktor.server.routing.Route
import io.ktor.server.routing.post
import java.util.UUID

/**
 * 聊天路由定义
 * 负责定义API路由和请求分发，遵循单一职责原则
 * 所有业务逻辑已拆分到专门的服务类中
 */

/**
 * 注册聊天相关的路由
 */
fun Route.chatRoutes() {
    val MNNChatService = MNNChatService()
    val logger = ChatLogger()

    post("/v1/chat/completions") {
        val traceId = UUID.randomUUID().toString()
        
        // 记录请求开始
        logger.logRequestStart(traceId, call)

        // 接收请求体
        val chatRequest = call.receive<OpenAIChatRequest>()
        
        // 委托给服务层处理
        MNNChatService.processChatCompletion(call, chatRequest, traceId)
    }
}