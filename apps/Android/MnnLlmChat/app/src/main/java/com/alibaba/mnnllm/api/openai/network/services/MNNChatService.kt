package com.alibaba.mnnllm.api.openai.network.services

import OpenAIChatRequest
import com.alibaba.mnnllm.android.MnnLlmApplication
import com.alibaba.mnnllm.android.llm.LlmSession
import com.alibaba.mnnllm.api.openai.di.ServiceLocator
import com.alibaba.mnnllm.api.openai.network.handlers.ResponseHandler
import com.alibaba.mnnllm.api.openai.network.logging.ChatLogger
import com.alibaba.mnnllm.api.openai.network.processors.MnnImageProcessor
import com.alibaba.mnnllm.api.openai.network.utils.MessageTransformer
import com.alibaba.mnnllm.api.openai.network.validators.ChatRequestValidator
import io.ktor.http.HttpStatusCode
import io.ktor.server.application.ApplicationCall
import io.ktor.server.response.respond
import kotlinx.coroutines.runBlocking

/**
 * 聊天服务
 * 负责协调各个组件，处理聊天请求的核心业务逻辑
 */
class MNNChatService {

    private val messageTransformer = MessageTransformer()
    private val responseHandler = ResponseHandler()
    private val chatRequestValidator = ChatRequestValidator()
    private val logger = ChatLogger()

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
     * @param call Ktor应用调用上下文
     * @param chatRequest 聊天请求对象
     * @param traceId 追踪ID
     */
    suspend fun processChatCompletion(
        call: ApplicationCall,
        chatRequest: OpenAIChatRequest,
        traceId: String
    ) {
        try {
            // 验证请求
            val validationResult = chatRequestValidator.validateChatRequest(chatRequest)
            if (!validationResult.isValid) {
                call.respond(
                    HttpStatusCode.BadRequest,
                    mapOf("error" to mapOf("message" to validationResult.errorMessage))
                )
                return
            }

            // 记录请求体
            logger.logRequestBody(traceId, chatRequest)

            // 转换为MNN格式的统一历史消息（包含系统提示词）
            val unifiedHistory = messageTransformer.convertToUnifiedMnnHistory(
                chatRequest.messages,
                imageProcessor,
                getLlmSession()
            )

            // 记录转换后的历史消息
            logger.logTransformedHistory(traceId, unifiedHistory)

            // 使用统一的完整历史消息进行推理（无状态模式）
            if (unifiedHistory.isNotEmpty()) {
                logger.logInferenceStart(traceId, unifiedHistory.size)

                if (chatRequest.stream == true) {
                    responseHandler.handleStreamResponseWithFullHistory(
                        call,
                        unifiedHistory,
                        traceId
                    )
                } else {
                    runBlocking {
                        responseHandler.handleNonStreamResponseWithFullHistory(call, unifiedHistory)
                    }
                }

                logger.logInferenceComplete(traceId)
            }
        } catch (e: Exception) {
            logger.logError(traceId, e, "请求失败")
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
}