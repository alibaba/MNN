package com.alibaba.mnnllm.android.chat.agent

import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.isActive
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.util.concurrent.atomic.AtomicReference

class AgentBlockingInference(
    private val submit: (AgentInferenceRequest, (String?) -> Boolean) -> Map<String, Any>,
    private val cancel: () -> Unit,
    private val dispatcher: CoroutineDispatcher = Dispatchers.IO,
    private val tokenCounter: (suspend (List<AgentMessage>) -> Int)? = null,
) : AgentInference {
    private val mutex = Mutex()

    override suspend fun generate(request: AgentInferenceRequest, onDelta: (String) -> Unit): AgentGeneration = mutex.withLock {
        coroutineScope {
            val context = currentCoroutineContext()
            val callbackError = AtomicReference<Exception?>()
            val submission = async(dispatcher) {
                currentCoroutineContext().ensureActive()
                val text = StringBuilder()
                val metrics = submit(request) { delta ->
                    if (!context.isActive || callbackError.get() != null) true
                    else if (delta == null) false
                    else if (delta.length > AgentOutputParser.MAX_OUTPUT_LENGTH - text.length) {
                        callbackError.set(IllegalArgumentException("Model output exceeds agent limit"))
                        true
                    } else {
                        text.append(delta)
                        try { onDelta(delta); !context.isActive }
                        catch (e: Exception) { callbackError.set(e); true }
                    }
                }
                currentCoroutineContext().ensureActive()
                callbackError.get()?.let { throw it }
                if (metrics["error"] == true || metrics["session_invalid"] == true || metrics["success"] == false) {
                    throw AgentInferenceException(metrics)
                }
                AgentGeneration(text.toString(), metrics = metrics)
            }
            try { submission.await() }
            catch (e: CancellationException) { cancel(); throw e }
        }
    }

    override suspend fun countTokens(messages: List<AgentMessage>): Int? = mutex.withLock {
        withContext(dispatcher) { tokenCounter?.invoke(messages) }
    }
}
