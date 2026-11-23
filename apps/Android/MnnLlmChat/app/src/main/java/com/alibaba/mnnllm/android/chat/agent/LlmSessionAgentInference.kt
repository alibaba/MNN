package com.alibaba.mnnllm.android.chat.agent

import android.util.Pair
import com.alibaba.mnnllm.android.llm.GenerateProgressListener
import com.alibaba.mnnllm.android.llm.LlmSession
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

internal interface AgentPassConfiguration {
    fun configure(request: AgentInferenceRequest)
    fun restore()
}

internal interface AgentSessionAccess {
    fun captureConfiguration(): AgentPassConfiguration
    fun submit(request: AgentInferenceRequest, onChunk: (String?) -> Boolean): Map<String, Any>
    fun cancel()
}

class LlmSessionAgentInference internal constructor(
    private val sessionAccess: AgentSessionAccess,
    private val preparePass: ((AgentInferenceRequest) -> Unit)? = null,
    private val dispatcher: CoroutineDispatcher = Dispatchers.IO,
    tokenCounter: (suspend (List<AgentMessage>) -> Int)? = null,
) : AgentInference {
    constructor(
        session: LlmSession,
        preparePass: ((AgentInferenceRequest) -> Unit)? = null,
        dispatcher: CoroutineDispatcher = Dispatchers.IO,
        tokenCounter: (suspend (List<AgentMessage>) -> Int)? = null,
    ) : this(NativeSessionAccess(session), preparePass, dispatcher, tokenCounter)

    private val mutex = Mutex()
    private val delegate = AgentBlockingInference(
        submit = { request, onChunk ->
            preparePass?.invoke(request)
            if (onChunk(null)) throw CancellationException("Agent pass cancelled before native submit")
            sessionAccess.submit(request, onChunk)
        },
        cancel = sessionAccess::cancel,
        dispatcher = dispatcher,
        tokenCounter = tokenCounter,
    )

    override suspend fun generate(request: AgentInferenceRequest, onDelta: (String) -> Unit): AgentGeneration = mutex.withLock {
        if (preparePass != null) return@withLock delegate.generate(request, onDelta)
        withContext(dispatcher) {
            val configuration = sessionAccess.captureConfiguration()
            try {
                currentCoroutineContext().ensureActive()
                configuration.configure(request)
                currentCoroutineContext().ensureActive()
                delegate.generate(request, onDelta)
            } finally {
                withContext(NonCancellable + dispatcher) { configuration.restore() }
            }
        }
    }

    override suspend fun countTokens(messages: List<AgentMessage>): Int? = mutex.withLock {
        delegate.countTokens(messages)
    }

    private class NativeSessionAccess(private val session: LlmSession) : AgentSessionAccess {
        override fun captureConfiguration(): AgentPassConfiguration {
            val snapshot = session.captureAgentConfig()
            return object : AgentPassConfiguration {
                override fun configure(request: AgentInferenceRequest) {
                    session.configureAgentPass(snapshot, request.outputTokenLimit, request.allowThinking)
                }
                override fun restore() { session.restoreAgentConfig(snapshot) }
            }
        }

        override fun submit(request: AgentInferenceRequest, onChunk: (String?) -> Boolean): Map<String, Any> =
            session.submitFullHistory(request.messages.map { Pair(it.role.wireName, it.content) },
                object : GenerateProgressListener {
                    override fun onProgress(progress: String?): Boolean = onChunk(progress)
                })

        override fun cancel() { session.cancelContextWork() }
    }
}
