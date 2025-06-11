package com.alibaba.mnnllm.api.openai.network.services

import OpenAIChatRequest
import com.alibaba.mnnllm.android.MnnLlmApplication
import com.alibaba.mnnllm.android.llm.LlmSession
import com.alibaba.mnnllm.api.openai.di.ServiceLocator
import com.alibaba.mnnllm.api.openai.network.handlers.ResponseHandler
import com.alibaba.mnnllm.api.openai.network.logging.ChatLogger
import com.alibaba.mnnllm.api.openai.network.processors.MnnImageProcessor
import com.alibaba.mnnllm.api.openai.network.queue.RequestQueueManager
import com.alibaba.mnnllm.api.openai.network.utils.MessageTransformer
import com.alibaba.mnnllm.api.openai.network.validators.ChatRequestValidator
import io.ktor.http.HttpStatusCode
import io.ktor.server.application.ApplicationCall
import io.ktor.server.response.respond
import kotlinx.coroutines.runBlocking
import java.util.UUID

/**
 * 聊天服务
 * 负责协调各个组件，处理聊天请求的核心业务逻辑
 * 现在所有请求都通过队列管理器进行排队处理，确保同一时刻只有一个LLM生成任务运行
 */
class MNNChatService {

    private val messageTransformer = MessageTransformer()
    private val responseHandler = ResponseHandler()
    private val chatRequestValidator = ChatRequestValidator()
    private val logger = ChatLogger()
    private val requestQueueManager = RequestQueueManager.getInstance()

    // 获取依赖
    private val chatSessionProvider = ServiceLocator.getChatSessionProvider()
    private val imageProcessor = MnnImageProcessor.getInstance(MnnLlmApplication.getAppContext())
    
    /**
     * 获取LLM会话实例
     * @return LlmSession实例，如果没有则抛出异常
     */
    private fun getLlmSession(): LlmSession {
        return chatSessionProvider.getLlmSession() 
            ?: throw IllegalStateException("No active LLM session available")
    }

    /**
     * 处理聊天完成请求
     * 现在所有请求都会被加入队列，确保按顺序处理
     * @param call Ktor应用调用上下文
     * @param chatRequest 聊天请求对象
     * @param traceId 追踪ID
     */
    suspend fun processChatCompletion(
        call: ApplicationCall,
        chatRequest: OpenAIChatRequest,
        traceId: String
    ) {
        // 生成唯一的请求ID
        val requestId = UUID.randomUUID().toString()
        
        // 先进行基本验证，避免无效请求进入队列
        val validationResult = chatRequestValidator.validateChatRequest(chatRequest)
        if (!validationResult.isValid) {
            call.respond(
                HttpStatusCode.BadRequest,
                mapOf("error" to mapOf("message" to validationResult.errorMessage))
            )
            return
        }
        
        logger.logRequestBody(traceId, chatRequest)
        logger.logInfo(traceId, "请求已提交到队列，requestId=$requestId")
        
        // 将请求提交到队列进行排队处理，并等待完成
        try {
            requestQueueManager.submitRequest(
                requestId = requestId,
                traceId = traceId,
                task = {
                    // 这里是实际的LLM处理逻辑
                    processLlmGeneration(call, chatRequest, traceId)
                },
                onComplete = {
                    logger.logInfo(traceId, "队列任务完成，requestId=$requestId")
                },
                onError = { exception ->
                    logger.logError(traceId, exception, "队列任务失败，requestId=$requestId")
                }
            )
        } catch (e: Exception) {
            logger.logError(traceId, e, "队列处理失败，requestId=$requestId")
            call.respond(
                HttpStatusCode.InternalServerError,
                mapOf(
                    "error" to mapOf(
                        "message" to "Queue processing failed",
                        "trace_id" to traceId
                    )
                )
            )
        }
    }
    
    /**
     * 实际的LLM生成处理逻辑
     * 这个方法会在队列中按顺序执行
     */
    private suspend fun processLlmGeneration(
        call: ApplicationCall,
        chatRequest: OpenAIChatRequest,
        traceId: String
    ) {
        try {
            // 转换为MNN格式的统一历史消息（包含系统提示词）
            val unifiedHistory = messageTransformer.convertToUnifiedMnnHistory(
                chatRequest.messages,
                imageProcessor,
                getLlmSession()
            )

            // 记录转换后的历史消息
            logger.logTransformedHistory(traceId, unifiedHistory)

            // 使用统一的完整历史消息进行推理（API服务模式）
            if (unifiedHistory.isNotEmpty()) {
                logger.logInferenceStart(traceId, unifiedHistory.size)

                if (chatRequest.stream == true) {
                    responseHandler.handleStreamResponseWithFullHistory(
                        call,
                        unifiedHistory,
                        traceId
                    )
                } else {
                    responseHandler.handleNonStreamResponseWithFullHistory(call, unifiedHistory)
                }

                logger.logInferenceComplete(traceId)
            }
        } catch (e: Exception) {
            logger.logError(traceId, e, "LLM生成失败")
            call.respond(
                HttpStatusCode.InternalServerError,
                mapOf(
                    "error" to mapOf(
                        "message" to "Internal server error",
                        "trace_id" to traceId
                    )
                )
            )
        }
    }
    
    /**
     * 获取队列统计信息
     */
    fun getQueueStats() = requestQueueManager.getQueueStats()
}