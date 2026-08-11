package com.alibaba.mnnllm.api.openai.network.services

import com.alibaba.mnnllm.android.MnnLlmApplication
import com.alibaba.mnnllm.android.llm.GenerateProgressListener
import com.alibaba.mnnllm.android.llm.LlmSession
import com.alibaba.mnnllm.api.openai.di.ServiceLocator
import com.alibaba.mnnllm.api.openai.network.compat.AnthropicAdapter
import com.alibaba.mnnllm.api.openai.network.compat.AnthropicMessagesRequest
import com.alibaba.mnnllm.api.openai.network.compat.AnthropicUsage
import com.alibaba.mnnllm.api.openai.network.logging.ChatLogger
import com.alibaba.mnnllm.api.openai.network.processors.MnnImageProcessor
import com.alibaba.mnnllm.api.openai.network.queue.RequestQueueManager
import com.alibaba.mnnllm.api.openai.network.utils.MessageTransformer
import com.alibaba.mnnllm.api.openai.network.validators.ChatRequestValidator
import io.ktor.http.ContentType
import io.ktor.http.HttpStatusCode
import io.ktor.server.application.ApplicationCall
import io.ktor.server.response.respond
import io.ktor.server.response.respondText
import io.ktor.server.sse.SSEServerContent
import io.ktor.server.sse.heartbeat
import io.ktor.sse.ServerSentEvent
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.util.UUID
import kotlin.time.Duration.Companion.seconds

class AnthropicMessagesService {
    private val messageTransformer = MessageTransformer()
    private val chatRequestValidator = ChatRequestValidator()
    private val logger = ChatLogger()
    private val requestQueueManager = RequestQueueManager.getInstance()
    private val chatSessionProvider = ServiceLocator.getChatSessionProvider()
    private val imageProcessor = MnnImageProcessor.getInstance(MnnLlmApplication.getAppContext())

    suspend fun processMessages(
        call: ApplicationCall,
        request: AnthropicMessagesRequest,
        traceId: String
    ) {
        val requestId = UUID.randomUUID().toString()
        val openAiRequest = AnthropicAdapter.toOpenAIRequest(request)

        val validationResult = chatRequestValidator.validateChatRequest(openAiRequest)
        if (!validationResult.isValid) {
            call.respond(
                HttpStatusCode.BadRequest,
                mapOf("error" to mapOf("message" to validationResult.errorMessage))
            )
            return
        }

        try {
            requestQueueManager.submitRequest(
                requestId = requestId,
                traceId = traceId,
                task = {
                    processGeneration(call, request, traceId)
                },
                onComplete = {
                    logger.logInfo(traceId, "Anthropic queue completed, requestId=$requestId")
                },
                onError = { exception ->
                    logger.logError(traceId, exception, "Anthropic queue failed, requestId=$requestId")
                }
            )
        } catch (e: Exception) {
            logger.logError(traceId, e, "Anthropic queue submit failed")
            call.respond(
                HttpStatusCode.InternalServerError,
                mapOf("error" to mapOf("message" to "Queue processing failed", "trace_id" to traceId))
            )
        }
    }

    private suspend fun processGeneration(
        call: ApplicationCall,
        request: AnthropicMessagesRequest,
        traceId: String
    ) {
        try {
            val llmSession = getLlmSession()
            if (llmSession == null) {
                logger.logInfo(traceId, "No active LLM session available")
                call.respond(
                    HttpStatusCode.ServiceUnavailable,
                    mapOf<String, Any>(
                        "error" to mapOf<String, String>(
                            "type" to "api_error",
                            "message" to "No active LLM session available. Please ensure a model is loaded."
                        )
                    )
                )
                return
            }

            val openAiRequest = AnthropicAdapter.toOpenAIRequest(request)
            val unifiedHistory = messageTransformer.convertToUnifiedMnnHistory(
                openAiRequest.messages,
                imageProcessor,
                llmSession
            )

            if (request.stream == true) {
                handleStreamResponse(call, unifiedHistory, request.model ?: "mnn-local", traceId, llmSession)
            } else {
                handleNonStreamResponse(call, unifiedHistory, request.model ?: "mnn-local", llmSession)
            }
        } catch (e: Exception) {
            logger.logError(traceId, e, "Anthropic generation failed")
            call.respond(
                HttpStatusCode.InternalServerError,
                mapOf<String, Any>(
                    "error" to mapOf<String, String>(
                        "type" to "api_error",
                        "message" to (e.message ?: "Internal server error")
                    )
                )
            )
        }
    }

    private suspend fun handleNonStreamResponse(
        call: ApplicationCall,
        history: List<android.util.Pair<String, String>>,
        model: String,
        llmSession: LlmSession
    ) {
        val fullResponse = StringBuilder()
        val result = llmSession.submitFullHistory(history, object : GenerateProgressListener {
            override fun onProgress(progress: String?): Boolean {
                progress?.let { fullResponse.append(it) }
                return false
            }
        })

        val usage = usageFromMetrics(result)
        val responseJson = AnthropicAdapter.createNonStreamResponse(
            responseId = "msg_${System.currentTimeMillis()}",
            model = model,
            text = fullResponse.toString(),
            usage = usage
        )

        call.respondText(responseJson, ContentType.Application.Json)
    }

    private suspend fun handleStreamResponse(
        call: ApplicationCall,
        history: List<android.util.Pair<String, String>>,
        model: String,
        traceId: String,
        llmSession: LlmSession
    ) {
        val responseId = "msg_${System.currentTimeMillis()}"

        call.respond(SSEServerContent(call) {
            heartbeat {
                period = 3.seconds
                event = ServerSentEvent(comments = "keep-alive")
            }

            send(ServerSentEvent(event = "message_start", data = "{\"type\":\"message_start\",\"message\":{\"id\":\"$responseId\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"$model\",\"content\":[],\"usage\":{\"input_tokens\":0,\"output_tokens\":0}}}"))
            send(ServerSentEvent(event = "content_block_start", data = "{\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}"))

            val channel = kotlinx.coroutines.channels.Channel<ServerSentEvent>(kotlinx.coroutines.channels.Channel.UNLIMITED)
            var metrics = hashMapOf<String, Any>()

            launch(kotlinx.coroutines.Dispatchers.IO) {
                try {
                    metrics = llmSession.submitFullHistory(history, object : GenerateProgressListener {
                        override fun onProgress(progress: String?): Boolean {
                            if (progress == null) {
                                return false
                            }
                            val safeText = progress
                                .replace("\\", "\\\\")
                                .replace("\"", "\\\"")
                                .replace("\n", "\\n")
                            val deltaData = "{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"$safeText\"}}"
                            channel.trySend(ServerSentEvent(event = "content_block_delta", data = deltaData))
                            return false
                        }
                    })
                } catch (e: Exception) {
                    logger.logError(traceId, e, "Anthropic stream generation failed")
                } finally {
                    val usage = usageFromMetrics(metrics)
                    val messageDelta = "{\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"input_tokens\":${usage.input_tokens},\"output_tokens\":${usage.output_tokens}}}"
                    channel.trySend(ServerSentEvent(event = "content_block_stop", data = "{\"type\":\"content_block_stop\",\"index\":0}"))
                    channel.trySend(ServerSentEvent(event = "message_delta", data = messageDelta))
                    channel.trySend(ServerSentEvent(event = "message_stop", data = "{\"type\":\"message_stop\"}"))
                    delay(50)
                    channel.close()
                }
            }

            for (event in channel) {
                send(event)
            }
        })
    }

    private fun usageFromMetrics(metrics: HashMap<String, Any>): AnthropicUsage {
        val promptTokens = (metrics["prompt_len"] as? Long)?.toInt() ?: 0
        val outputTokens = (metrics["decode_len"] as? Long)?.toInt() ?: 0
        return AnthropicUsage(input_tokens = promptTokens, output_tokens = outputTokens)
    }

    private fun getLlmSession(): LlmSession? {
        return chatSessionProvider.getLlmSession()
    }
}
