package com.alibaba.mnnllm.api.openai.runtime

import android.os.Looper
import com.alibaba.mnnllm.android.chat.model.ChatDataItem
import com.alibaba.mnnllm.android.llm.ChatService
import com.alibaba.mnnllm.android.llm.LlmSession
import com.alibaba.mnnllm.android.model.ModelUtils
import com.alibaba.mnnllm.android.modelsettings.ModelConfig
import kotlinx.coroutines.CancellationException
import timber.log.Timber

object DefaultLlmRuntimeController : LlmRuntimeController {
    private val lock = Any()
    private var activeModelId: String? = null
    private var activeSession: LlmSession? = null
    private var activeUseAppConfig: Boolean = false
    private var activeOwner: Any? = null

    override fun ensureSession(
        modelId: String,
        forceReload: Boolean,
        useAppConfig: Boolean,
        configPath: String?,
        sessionId: String?,
        historyList: List<ChatDataItem>?,
        deferLoad: Boolean,
        owner: Any?
    ): EnsureSessionResult {
        var createdSession: LlmSession? = null
        var previousSession: LlmSession? = null
        try {
            val session = synchronized(lock) {
                val current = activeSession
                if (current != null && RuntimeSessionReusePolicy.shouldReuse(
                        forceReload, activeModelId, modelId, current.isModelLoaded(),
                        activeUseAppConfig, useAppConfig
                    )) {
                    if (owner != null) activeOwner = owner
                    return EnsureSessionResult(true, current, modelId)
                }
                check(current == null || Looper.myLooper() != Looper.getMainLooper()) {
                    "Model replacement must run off the main thread"
                }
                val resolvedConfigPath = configPath ?: if (useAppConfig) {
                    ModelUtils.getConfigPathForModel(modelId)
                } else {
                    ModelConfig.getDefaultConfigFile(modelId)
                } ?: return EnsureSessionResult(false, modelId = modelId, reason = "MODEL_CONFIG_NOT_FOUND")
                val chatSession = ChatService.provide().createSession(
                    modelId = modelId,
                    modelName = ModelUtils.getModelName(modelId) ?: modelId,
                    sessionIdParam = sessionId ?: "service_runtime_${System.currentTimeMillis()}",
                    historyList = historyList,
                    configPath = resolvedConfigPath,
                    useNewConfig = !useAppConfig,
                    useCustomConfig = useAppConfig
                )
                val llmSession = chatSession as? LlmSession
                if (llmSession == null) {
                    chatSession.release()
                    return EnsureSessionResult(false, modelId = modelId, reason = "SESSION_NOT_LLM")
                }
                createdSession = llmSession
                llmSession.setKeepHistory(true)
                previousSession = activeSession
                activeSession = llmSession
                activeModelId = modelId
                activeUseAppConfig = useAppConfig
                activeOwner = owner
                llmSession
            }
            previousSession?.release()
            if (!deferLoad) session.load()
            return synchronized(lock) {
                if (activeSession !== session || (!deferLoad && !session.isModelLoaded())) {
                    EnsureSessionResult(false, modelId = modelId, reason = "SESSION_REPLACED")
                } else {
                    EnsureSessionResult(true, session, modelId)
                }
            }
        } catch (e: CancellationException) {
            createdSession?.let { releaseSession(it, owner) }
            throw e
        } catch (e: OutOfMemoryError) {
            createdSession?.let { releaseSession(it, owner) }
            throw e
        } catch (e: Exception) {
            createdSession?.let { releaseSession(it, owner) }
            Timber.w(e, "ensureSession failed for modelId=%s", modelId)
            return EnsureSessionResult(false, modelId = modelId, reason = e.message ?: "SESSION_INIT_FAILED")
        }
    }

    override fun getActiveSession(): LlmSession? = synchronized(lock) {
        activeSession?.takeIf { RuntimeSessionReusePolicy.shouldExposeActiveSession(it.isModelLoaded()) }
    }

    override fun getActiveModelId(): String? = synchronized(lock) { activeModelId }

    override fun getThinkingEnabled(): Boolean? {
        val modelId = getActiveModelId() ?: return null
        val config = ModelConfig.loadConfig(modelId) ?: return null
        return config.jinja?.context?.enableThinking != false
    }

    override fun setThinkingEnabled(enabled: Boolean): Boolean {
        val session = getActiveSession() ?: return false
        return try {
            session.updateThinking(enabled)
            true
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            Timber.w(e, "setThinkingEnabled failed enabled=%s", enabled)
            false
        }
    }

    override fun releaseSession(expectedSession: LlmSession?, owner: Any?) {
        val session = synchronized(lock) {
            if (expectedSession != null && activeSession !== expectedSession) return
            if (owner != null && activeOwner !== owner) return
            activeSession.also {
                activeSession = null
                activeModelId = null
                activeUseAppConfig = false
                activeOwner = null
            }
        }
        session?.release()
    }
}
