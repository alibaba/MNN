package com.alibaba.mnnllm.api.openai.runtime

import com.alibaba.mnnllm.android.chat.model.ChatDataItem
import com.alibaba.mnnllm.android.llm.ChatService
import com.alibaba.mnnllm.android.llm.LlmSession
import com.alibaba.mnnllm.android.model.ModelUtils
import com.alibaba.mnnllm.android.modelsettings.ModelConfig
import timber.log.Timber

object DefaultLlmRuntimeController : LlmRuntimeController {
    private val lock = Any()
    private var activeModelId: String? = null
    private var activeSession: LlmSession? = null
    private var activeUseAppConfig: Boolean = false

    override fun ensureSession(
        modelId: String,
        forceReload: Boolean,
        useAppConfig: Boolean,
        configPath: String?,
        sessionId: String?,
        historyList: List<ChatDataItem>?,
        deferLoad: Boolean
    ): EnsureSessionResult {
        synchronized(lock) {
            val currentSession = activeSession
            if (
                currentSession != null &&
                RuntimeSessionReusePolicy.shouldReuse(
                    forceReload = forceReload,
                    activeModelId = activeModelId,
                    requestedModelId = modelId,
                    isSessionLoaded = currentSession.isModelLoaded(),
                    activeUseAppConfig = activeUseAppConfig,
                    requestedUseAppConfig = useAppConfig
                )
            ) {
                return EnsureSessionResult(
                    success = true,
                    session = currentSession,
                    modelId = modelId
                )
            }

            if (currentSession != null) {
                releaseSessionLocked()
            }

            val resolvedConfigPath = configPath ?: if (useAppConfig) {
                ModelUtils.getConfigPathForModel(modelId)
            } else {
                ModelConfig.getDefaultConfigFile(modelId)
            } ?: return EnsureSessionResult(
                success = false,
                modelId = modelId,
                reason = "MODEL_CONFIG_NOT_FOUND"
            )

            val modelName = ModelUtils.getModelName(modelId) ?: modelId
            val resolvedSessionId = sessionId ?: "service_runtime_${System.currentTimeMillis()}"

            return runCatching {
                val chatSession = ChatService.provide().createSession(
                    modelId = modelId,
                    modelName = modelName,
                    sessionIdParam = resolvedSessionId,
                    historyList = historyList,
                    configPath = resolvedConfigPath,
                    useNewConfig = !useAppConfig,
                    useCustomConfig = useAppConfig
                )

                val llmSession = chatSession as? LlmSession
                    ?: return EnsureSessionResult(
                        success = false,
                        modelId = modelId,
                        reason = "SESSION_NOT_LLM"
                    )

                llmSession.setKeepHistory(true)
                if (!deferLoad) {
                    llmSession.load()
                }
                activeSession = llmSession
                activeModelId = modelId
                activeUseAppConfig = useAppConfig
                EnsureSessionResult(
                    success = true,
                    session = llmSession,
                    modelId = modelId
                )
            }.getOrElse { error ->
                Timber.w(error, "ensureSession failed for modelId=%s", modelId)
                releaseSessionLocked()
                EnsureSessionResult(
                    success = false,
                    modelId = modelId,
                    reason = "SESSION_INIT_FAILED"
                )
            }
        }
    }

    override fun getActiveSession(): LlmSession? {
        synchronized(lock) {
            val session = activeSession ?: return null
            return session.takeIf {
                RuntimeSessionReusePolicy.shouldExposeActiveSession(it.isModelLoaded())
            }
        }
    }

    override fun getActiveModelId(): String? {
        synchronized(lock) {
            return activeModelId
        }
    }

    override fun getThinkingEnabled(): Boolean? {
        synchronized(lock) {
            val modelId = activeModelId ?: return null
            val config = ModelConfig.loadConfig(modelId) ?: return null
            return config.jinja?.context?.enableThinking != false
        }
    }

    override fun setThinkingEnabled(enabled: Boolean): Boolean {
        synchronized(lock) {
            val session = getActiveSession() ?: return false
            return runCatching {
                session.updateThinking(enabled)
                true
            }.getOrElse { error ->
                Timber.w(error, "setThinkingEnabled failed enabled=%s", enabled)
                false
            }
        }
    }

    override fun releaseSession() {
        synchronized(lock) {
            releaseSessionLocked()
        }
    }

    private fun releaseSessionLocked() {
        runCatching {
            activeSession?.release()
        }.onFailure { error ->
            Timber.w(error, "releaseSession failed")
        }
        activeSession = null
        activeModelId = null
        activeUseAppConfig = false
    }
}
