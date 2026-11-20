package com.alibaba.mnnllm.api.openai.network.handlers

import com.alibaba.mnnllm.android.llm.GenerateProgressListener
import com.alibaba.mnnllm.android.llm.LlmSession
import com.alibaba.mnnllm.api.openai.di.ServiceLocator
import com.alibaba.mnnllm.api.openai.network.logging.ChatLogger
import com.alibaba.mnnllm.api.openai.network.utils.ChatResponseFormatter
import com.alibaba.mnnllm.api.openai.network.utils.LlmGenerationException
import com.alibaba.mnnllm.api.openai.network.utils.LlmResultBoundary
import io.ktor.server.application.ApplicationCall
import io.ktor.server.response.respond
import io.ktor.server.response.respondText
import io.ktor.server.sse.SSEServerContent
import io.ktor.server.sse.heartbeat
import io.ktor.sse.ServerSentEvent
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlin.time.Duration.Companion.seconds

class ResponseHandler {
    private val chatResponseFormatter = ChatResponseFormatter()
    private val logger = ChatLogger()
    private val chatSessionProvider = ServiceLocator.getChatSessionProvider()

    companion object {
        private const val HEARTBEAT_PERIOD_SECONDS = 3L
        private const val DONE_DELAY_MS = 100L
    }

    suspend fun handleStreamResponseWithFullHistory(
        call: ApplicationCall,
        history: List<android.util.Pair<String, String>>,
        traceId: String
    ) {
        val responseMetadata = createResponseMetadata()
        call.respond(SSEServerContent(call) {
            heartbeat {
                period = HEARTBEAT_PERIOD_SECONDS.seconds
                event = ServerSentEvent(comments = "keep-alive")
            }
            val channel = Channel<ServerSentEvent>(Channel.UNLIMITED)
            val producer = launch(Dispatchers.IO) {
                try {
                    val llmSession = getLlmSession() ?: throw LlmGenerationException("LlmSession not available")
                    val context = currentCoroutineContext()
                    val result = LlmResultBoundary.submitFullHistory(llmSession, history, object : GenerateProgressListener {
                        override fun onProgress(progress: String?): Boolean {
                            if (!context.isActive) return true
                            if (progress == null) return false
                            logger.logStreamDelta(traceId, progress)
                            val json = chatResponseFormatter.createDeltaResponse(
                                responseMetadata.responseId, responseMetadata.created, progress, false
                            )
                            return channel.trySend(ServerSentEvent(data = json)).isFailure
                        }
                    })
                    context.ensureActive()
                    val finalJson = chatResponseFormatter.createDeltaResponse(
                        responseMetadata.responseId,
                        responseMetadata.created,
                        "",
                        true,
                        usage = chatResponseFormatter.createUsageFromMetrics(result)
                    )
                    channel.send(ServerSentEvent(data = finalJson))
                    delay(DONE_DELAY_MS)
                    channel.send(ServerSentEvent(data = "[DONE]"))
                    logger.logStreamEnd(traceId)
                    logger.logInferenceComplete(traceId)
                } catch (e: CancellationException) {
                    throw e
                } catch (e: Exception) {
                    logger.logError(traceId, e, "Error during generation")
                    channel.trySend(ServerSentEvent(event = "error", data = LlmResultBoundary.errorJson(e)))
                } finally {
                    channel.close()
                }
            }
            try {
                for (event in channel) send(event)
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                logger.logError(traceId, e, "Error sending SSE event")
            } finally {
                channel.cancel()
                producer.cancel()
            }
        })
    }

    suspend fun handleNonStreamResponseWithFullHistory(
        call: ApplicationCall,
        history: List<android.util.Pair<String, String>>
    ) {
        val fullResponse = StringBuilder()
        val responseMetadata = createResponseMetadata()
        val result = withContext(Dispatchers.IO) {
            val llmSession = getLlmSession() ?: throw LlmGenerationException("LlmSession not available")
            val context = currentCoroutineContext()
            LlmResultBoundary.submitFullHistory(llmSession, history, object : GenerateProgressListener {
                override fun onProgress(progress: String?): Boolean {
                    if (!context.isActive) return true
                    progress?.let { fullResponse.append(it) }
                    return false
                }
            }).also { context.ensureActive() }
        }
        val response = createNonStreamResponse(responseMetadata, fullResponse.toString(), result)
        call.respondText(response, io.ktor.http.ContentType.Application.Json)
    }

    private data class ResponseMetadata(val responseId: String, val created: Long)

    private fun createResponseMetadata(): ResponseMetadata {
        val timestamp = System.currentTimeMillis()
        return ResponseMetadata(responseId = "chatcmpl-$timestamp", created = timestamp / 1000)
    }

    private fun getLlmSession(): LlmSession? = chatSessionProvider.getLlmSession()

    private fun createNonStreamResponse(
        responseMetadata: ResponseMetadata,
        content: String,
        metrics: HashMap<String, Any>
    ): String {
        val usage = chatResponseFormatter.createUsageFromMetrics(metrics)
        return chatResponseFormatter.createChatCompletionResponse(responseMetadata.responseId, responseMetadata.created, content, usage)
    }
}
