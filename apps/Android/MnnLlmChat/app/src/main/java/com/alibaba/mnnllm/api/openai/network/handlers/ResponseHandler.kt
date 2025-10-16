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

/** * response processoror * responsible for processingstreamingand non-streaming responsegenerateand sending * * mainfunctionality： * - streaming response processor（SSE） * - non-streaming response processor * - unifiederrorprocessand logrecording * - LLM sessionmanaging*/
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

    /** * processwithhistorymessagestreamingresponse * * @param call Ktorapplicationcallcontext * @param history historymessagelist * @param traceId traceID*/
    suspend fun handleStreamResponseWithFullHistory(
        call: ApplicationCall,
        history: List<android.util.Pair<String, String>>,
        traceId: String
    ) {
        val responseMetadata = createResponseMetadata()
        var isFinished = false

        call.respond(SSEServerContent(call) {
            //set heartbeenat
            heartbeat {
                period = HEARTBEAT_PERIOD_SECONDS.seconds
                event = ServerSentEvent(comments = "keep-alive")
            }
            
            launch {
                try {
                    val llmSession = getLlmSession()
                    if (llmSession != null) {
                        //processstreaminggenerate
                        val result = llmSession.submitFullHistory(history, object : GenerateProgressListener {
                            override fun onProgress(progress: String?): Boolean {
                                return try {
                                    when {
                                        progress != null -> {
                                            //sendprogresstoken
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
                                            false //continuegenerate
                                        }
                                        else -> {
                                            //generatecomplete，butnotatheresendcompletesignal
                                            //completion signalwillat submitFullHistoryreturn then send
                                            isFinished = true
                                            logger.logStreamEnd(traceId)
                                            false
                                        }
                                    }
                                } catch (e: Exception) {
                                    //checkwhetherasconnectcloseexception
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
                                    true //cancelgenerate
                                }
                            }
                        })
                        
                        //generation completethen sendfinalchunk (containingusagestatisticsinfo)
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
                                
                                //send[DONE]marker
                                delay(DONE_DELAY_MS)
                                try {
                                    send(ServerSentEvent(data = "[DONE]"))
                                } catch (e: Exception) {
                                    logger.logWarning(traceId, "连接已关闭，无法发送[DONE]标记", e)
                                }
                            } catch (e: Exception) {
                                //checkwhetherasconnectcloseexception
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
                        //processLLMsessionerror
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
                    //checkwhetherasconnectcloseexception
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
                            //checkwhetherasconnectcloseexception
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

    /** * processwithhistorymessagenon-streamingresponse * * @param call Ktorapplicationcallcontext * @param history historymessagelist*/
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
        //set correctContent-Typeas application/json
        call.response.headers.append("Content-Type", "application/json")
        call.respondText(response, io.ktor.http.ContentType.Application.Json)
    }
    
    // ========================
    //private helpermethod
    // ========================
    
    /** * response metadata*/
    private data class ResponseMetadata(
        val responseId: String,
        val created: Long
    )
    
    /** * createresponse metadata*/
    private fun createResponseMetadata(): ResponseMetadata {
        val timestamp = System.currentTimeMillis()
        return ResponseMetadata(
            responseId = "chatcmpl-$timestamp",
            created = timestamp / 1000
        )
    }
    
    /**
     * getLLMsession*/
    private fun getLlmSession(): LlmSession? {
        return chatSessionProvider.getLlmSession()
    }
    

    
    /** * processnon-streaminggenerate*/
    private fun processNonStreamGeneration(
        llmSession: LlmSession,
        history: List<android.util.Pair<String, String>>,
        fullResponse: StringBuilder
    ): HashMap<String, Any> {
        return llmSession.submitFullHistory(history, object : GenerateProgressListener {
            override fun onProgress(progress: String?): Boolean {
                progress?.let { fullResponse.append(it) }
                return false //continuegenerate
            }
        })
    }
    
    /** * createnon-streamingresponse*/
    private fun createNonStreamResponse(
        responseMetadata: ResponseMetadata,
        content: String,
        metrics: HashMap<String, Any>
    ): String {
        val usage = chatResponseFormatter.createUsageFromMetrics(metrics)
        return chatResponseFormatter.createChatCompletionResponse(responseMetadata.responseId, responseMetadata.created, content, usage)
    }
}