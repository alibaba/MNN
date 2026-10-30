package com.alibaba.mnnllm.api.openai.network.services

import com.alibaba.mnnllm.android.MnnLlmApplication
import com.alibaba.mnnllm.android.llm.GenerateProgressListener
import com.alibaba.mnnllm.api.openai.di.ServiceLocator
import com.alibaba.mnnllm.api.openai.network.compat.OllamaAdapter
import com.alibaba.mnnllm.api.openai.network.processors.MnnImageProcessor
import com.alibaba.mnnllm.api.openai.network.queue.RequestQueueManager
import com.alibaba.mnnllm.api.openai.network.utils.MessageTransformer
import io.ktor.http.ContentType
import io.ktor.http.HttpStatusCode
import io.ktor.server.application.ApplicationCall
import io.ktor.server.response.respondText
import io.ktor.server.response.respondTextWriter
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.channels.ClosedSendChannelException
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import java.util.UUID

class OllamaService {
    suspend fun generate(call: ApplicationCall, request: OllamaAdapter.Request, chat: Boolean, traceId: String) {
        try {
            RequestQueueManager.getInstance().submitRequest(UUID.randomUUID().toString(), traceId, task = {
                call.coroutineContext.ensureActive()
                val session = ServiceLocator.getChatSessionProvider().getLlmSession()
                if (session == null) {
                    error(call, HttpStatusCode.ServiceUnavailable, "No active model available")
                    return@submitRequest
                }
                if (request.model != session.modelId()) {
                    error(call, HttpStatusCode.NotFound, "Model not loaded: ${request.model}")
                    return@submitRequest
                }
                val history = MessageTransformer().convertToUnifiedMnnHistory(
                    request.messages, MnnImageProcessor.getInstance(MnnLlmApplication.getAppContext()), session
                )
                if (!request.stream) {
                    val text = StringBuilder()
                    val metrics = session.submitFullHistory(history, object : GenerateProgressListener {
                        override fun onProgress(progress: String?): Boolean { if (progress != null) text.append(progress); return false }
                    })
                    OllamaAdapter.checkGenerationResult(metrics)
                    call.respondText(response(request.model, text.toString(), true, chat, metrics), ContentType.Application.Json)
                } else {
                    call.respondTextWriter(contentType = ContentType.parse("application/x-ndjson")) {
                        coroutineScope {
                            val channel = Channel<String>(Channel.BUFFERED)
                            val producer = launch(Dispatchers.IO) {
                                try {
                                    val metrics = session.submitFullHistory(history, object : GenerateProgressListener {
                                        override fun onProgress(progress: String?): Boolean {
                                            if (progress == null) return false
                                            if (!call.coroutineContext.isActive) return true
                                            val chunk = response(request.model, progress, false, chat)
                                            val result = channel.trySend(chunk)
                                            if (result.isSuccess) return false
                                            if (result.isClosed) return true
                                            return try {
                                                runBlocking { channel.send(chunk) }
                                                false
                                            } catch (_: CancellationException) {
                                                true
                                            } catch (_: ClosedSendChannelException) {
                                                true
                                            }
                                        }
                                    })
                                    OllamaAdapter.checkGenerationResult(metrics)
                                    channel.send(response(request.model, "", true, chat, metrics))
                                } catch (e: CancellationException) {
                                    throw e
                                } catch (e: Exception) {
                                    channel.trySend(buildJsonObject { put("error", e.message ?: "Generation failed") }.toString())
                                } finally {
                                    channel.close()
                                }
                            }
                            try {
                                for (line in channel) { write(line); write("\n"); flush() }
                            } finally {
                                channel.cancel()
                                producer.cancel()
                            }
                        }
                    }
                }
            })
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            if (!call.response.isCommitted) {
                error(call, HttpStatusCode.InternalServerError, "Generation failed")
            }
        }
    }

    private fun response(model: String, text: String, done: Boolean, chat: Boolean, metrics: Map<String, Any> = emptyMap()): String =
        OllamaAdapter.response(model, text, done, chat, (metrics["prompt_len"] as? Number)?.toInt() ?: 0,
            (metrics["decode_len"] as? Number)?.toInt() ?: 0)

    private suspend fun error(call: ApplicationCall, status: HttpStatusCode, message: String) {
        call.respondText(buildJsonObject { put("error", message) }.toString(), ContentType.Application.Json, status)
    }
}
