package com.alibaba.mnnllm.api.openai.network.handlers

import com.alibaba.mnnllm.android.llm.GenerateProgressListener
import com.alibaba.mnnllm.android.llm.LlmSession
import com.alibaba.mnnllm.api.openai.di.ServiceLocator
import com.alibaba.mnnllm.api.openai.network.logging.ChatLogger
import com.alibaba.mnnllm.api.openai.network.utils.ChatResponseFormatter
import io.ktor.http.HttpStatusCode
import io.ktor.server.application.ApplicationCall
import io.ktor.server.response.respond
import io.ktor.server.response.respondText
import io.ktor.server.sse.SSEServerContent
import io.ktor.server.sse.heartbeat
import io.ktor.sse.ServerSentEvent
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlin.time.Duration.Companion.seconds

/**
 * 响应处理器
 * 负责处理流式和非流式响应的生成和发送
 * 
 * 主要功能：
 * - 流式响应处理（SSE）
 * - 非流式响应处理
 * - 统一的错误处理和日志记录
 * - LLM会话管理
 */
class ResponseHandler {
    
    private val chatResponseFormatter = ChatResponseFormatter()
    private val logger = ChatLogger()
    private val chatSessionProvider = ServiceLocator.getChatSessionProvider()
    
    companion object {
        private const val HEARTBEAT_PERIOD_SECONDS = 3L
        private const val DONE_DELAY_MS = 100L
        private const val MODEL_NAME = "mnn-llm"
        private const val COMPLETION_CHUNK_OBJECT = "chat.completion.chunk"
        private const val COMPLETION_OBJECT = "chat.completion"
    }

    /**
     * 处理带历史消息的流式响应
     * 
     * @param call Ktor应用调用上下文
     * @param history 历史消息列表
     * @param traceId 追踪ID
     */
    suspend fun handleStreamResponseWithFullHistory(
        call: ApplicationCall,
        history: List<android.util.Pair<String, String>>,
        traceId: String
    ) {
        val responseMetadata = createResponseMetadata()
        var isFinished = false

        call.respond(SSEServerContent(call) {
            // 设置心跳
            heartbeat {
                period = HEARTBEAT_PERIOD_SECONDS.seconds
                event = ServerSentEvent(comments = "keep-alive")
            }
            
            launch {
                try {
                    val llmSession = getLlmSession()
                    if (llmSession != null) {
                        // 处理流式生成
                        val result = llmSession.submitFullHistory(history, object : GenerateProgressListener {
                            override fun onProgress(progress: String?): Boolean {
                                return try {
                                    when {
                                        progress != null -> {
                                            // 发送进度token
                                            logger.logStreamDelta(traceId, progress)
                                            val json = chatResponseFormatter.createDeltaResponse(
                                                responseMetadata.responseId,
                                                responseMetadata.created,
                                                progress,
                                                false
                                            )
                                            runBlocking {
                                                send(ServerSentEvent(data = json))
                                            }
                                            false // 继续生成
                                        }
                                        else -> {
                                            // 生成完成，但不在这里发送完成信号
                                            // 完成信号将在submitFullHistory返回后发送
                                            isFinished = true
                                            logger.logStreamEnd(traceId)
                                            false
                                        }
                                    }
                                } catch (e: Exception) {
                                    // 检查是否为连接关闭异常
                                    val isConnectionClosed = e is io.ktor.utils.io.ClosedWriteChannelException ||
                                            e.cause is io.ktor.utils.io.ClosedWriteChannelException ||
                                            e.cause is io.ktor.util.cio.ChannelWriteException ||
                                            e.message?.contains("Cannot write to channel") == true ||
                                            e.message?.contains("ClosedChannelException") == true
                                    
                                    if (isConnectionClosed) {
                                        logger.logWarning(traceId, "客户端连接已断开，停止生成", e)
                                    } else {
                                        logger.logError(traceId, e, "发送SSE事件时出错")
                                    }
                                    true // 取消生成
                                }
                            }
                        })
                        
                        // 生成完成后发送最终chunk（包含usage统计信息）
                        if (isFinished) {
                            try {
                                val usage = chatResponseFormatter.createUsageFromMetrics(result)
                                val finalJson = chatResponseFormatter.createDeltaResponse(
                                    responseMetadata.responseId,
                                    responseMetadata.created,
                                    "",
                                    true,
                                    usage = usage
                                )
                                send(ServerSentEvent(data = finalJson))
                                
                                // 发送[DONE]标记
                                delay(DONE_DELAY_MS)
                                try {
                                    send(ServerSentEvent(data = "[DONE]"))
                                } catch (e: Exception) {
                                    logger.logWarning(traceId, "连接已关闭，无法发送[DONE]标记", e)
                                }
                            } catch (e: Exception) {
                                // 检查是否为连接关闭异常
                                val isConnectionClosed = e is io.ktor.utils.io.ClosedWriteChannelException ||
                                        e.cause is io.ktor.utils.io.ClosedWriteChannelException ||
                                        e.cause is io.ktor.util.cio.ChannelWriteException ||
                                        e.message?.contains("Cannot write to channel") == true ||
                                        e.message?.contains("ClosedChannelException") == true
                                
                                if (isConnectionClosed) {
                                    logger.logWarning(traceId, "客户端连接已断开，跳过最终响应发送", e)
                                } else {
                                    logger.logError(traceId, e, "发送最终响应时出错")
                                }
                            }
                            close()
                        }
                        logger.logInferenceComplete(traceId)
                    } else {
                        // 处理LLM会话错误
                        logger.logLlmSessionError(traceId)
                        val errorJson = chatResponseFormatter.createDeltaResponse(
                            responseMetadata.responseId,
                            responseMetadata.created,
                            "LlmSession not available",
                            true
                        )
                        send(ServerSentEvent(data = errorJson))
                        close()
                    }
                } catch (e: Exception) {
                    // 检查是否为连接关闭异常
                    val isConnectionClosed = e is io.ktor.utils.io.ClosedWriteChannelException ||
                            e.cause is io.ktor.utils.io.ClosedWriteChannelException ||
                            e.cause is io.ktor.util.cio.ChannelWriteException ||
                            e.message?.contains("Cannot write to channel") == true ||
                            e.message?.contains("ClosedChannelException") == true
                    
                    if (isConnectionClosed) {
                        logger.logWarning(traceId, "客户端连接已断开，停止处理", e)
                    } else {
                        logger.logError(traceId, e, "生成文本时出错")
                    }
                    
                    if (!isFinished) {
                        try {
                            val finalJson = chatResponseFormatter.createDeltaResponse(
                                responseMetadata.responseId,
                                responseMetadata.created,
                                "",
                                true
                            )
                            send(ServerSentEvent(data = finalJson))
                            
                            delay(DONE_DELAY_MS)
                            try {
                                send(ServerSentEvent(data = "[DONE]"))
                            } catch (doneException: Exception) {
                                logger.logWarning(traceId, "连接已关闭，无法发送[DONE]标记", doneException)
                            }
                        } catch (closeException: Exception) {
                            // 检查是否为连接关闭异常
                            val isCloseConnectionClosed = closeException is io.ktor.utils.io.ClosedWriteChannelException ||
                                    closeException.cause is io.ktor.utils.io.ClosedWriteChannelException ||
                                    closeException.cause is io.ktor.util.cio.ChannelWriteException ||
                                    closeException.message?.contains("Cannot write to channel") == true ||
                                    closeException.message?.contains("ClosedChannelException") == true
                            
                            if (isCloseConnectionClosed) {
                                logger.logWarning(traceId, "客户端连接已断开，跳过错误结束发送", closeException)
                            } else {
                                logger.logError(traceId, closeException, "发送错误结束时出错")
                            }
                        }
                        close()
                    }
                }
            }
        })
    }

    /**
     * 处理带历史消息的非流式响应
     * 
     * @param call Ktor应用调用上下文
     * @param history 历史消息列表
     */
    suspend fun handleNonStreamResponseWithFullHistory(
        call: ApplicationCall,
        history: List<android.util.Pair<String, String>>
    ) {
        val fullResponse = StringBuilder()
        val responseMetadata = createResponseMetadata()
        
        var result = HashMap<String, Any>()
        try {
            val llmSession = getLlmSession()
            if (llmSession != null) {
                result = processNonStreamGeneration(llmSession, history, fullResponse)
            } else {
                fullResponse.append("Error: LlmSession not available")
            }
        } catch (e: Exception) {
            fullResponse.append("Error: ${e.message}")
        }

        val response = createNonStreamResponse(responseMetadata, fullResponse.toString(), result)
        // 设置正确的Content-Type为application/json
        call.response.headers.append("Content-Type", "application/json")
        call.respondText(response, io.ktor.http.ContentType.Application.Json)
    }
    
    // ========================
    // 私有辅助方法
    // ========================
    
    /**
     * 响应元数据
     */
    private data class ResponseMetadata(
        val responseId: String,
        val created: Long
    )
    
    /**
     * 创建响应元数据
     */
    private fun createResponseMetadata(): ResponseMetadata {
        val timestamp = System.currentTimeMillis()
        return ResponseMetadata(
            responseId = "chatcmpl-$timestamp",
            created = timestamp / 1000
        )
    }
    
    /**
     * 获取LLM会话
     */
    private fun getLlmSession(): LlmSession? {
        return chatSessionProvider.getLlmSession()
    }
    

    
    /**
     * 处理非流式生成
     */
    private fun processNonStreamGeneration(
        llmSession: LlmSession,
        history: List<android.util.Pair<String, String>>,
        fullResponse: StringBuilder
    ): HashMap<String, Any> {
        return llmSession.submitFullHistory(history, object : GenerateProgressListener {
            override fun onProgress(progress: String?): Boolean {
                progress?.let { fullResponse.append(it) }
                return false // 继续生成
            }
        })
    }
    
    /**
     * 创建非流式响应
     */
    private fun createNonStreamResponse(
        responseMetadata: ResponseMetadata,
        content: String,
        metrics: HashMap<String, Any>
    ): String {
        val usage = chatResponseFormatter.createUsageFromMetrics(metrics)
        return chatResponseFormatter.createChatCompletionResponse(responseMetadata.responseId, responseMetadata.created, content, usage)
    }
}