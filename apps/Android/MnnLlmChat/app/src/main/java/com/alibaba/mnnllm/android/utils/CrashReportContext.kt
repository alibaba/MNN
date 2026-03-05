package com.alibaba.mnnllm.android.utils

import com.google.firebase.crashlytics.FirebaseCrashlytics

/**
 * Centralized Crashlytics context updater so crash reports always include
 * latest UI and model execution state.
 */
object CrashReportContext {
    private const val KEY_CURRENT_ACTIVITY = "current_activity"
    private const val KEY_MODEL_ID = "model_id"
    private const val KEY_SESSION_ID = "session_id"
    private const val KEY_LLM_SET_CONFIG_STAGE = "llm_set_config_stage"
    private const val KEY_LLM_SET_CONFIG_SOURCE_LEN = "llm_set_config_source_len"
    private const val KEY_LLM_SET_CONFIG_TRUNCATED = "llm_set_config_truncated"
    private const val MAX_LLM_CONFIG_LOG_CHARS = 3000

    fun setCurrentActivity(activityName: String?) {
        updateSafely {
            val value = activityName?.takeIf { it.isNotBlank() } ?: "none"
            setCustomKey(KEY_CURRENT_ACTIVITY, value)
            log("activity=$value")
        }
    }

    fun setCurrentModel(modelId: String?, sessionId: String?) {
        updateSafely {
            val modelValue = modelId?.takeIf { it.isNotBlank() } ?: "none"
            val sessionValue = sessionId?.takeIf { it.isNotBlank() } ?: "none"
            setCustomKey(KEY_MODEL_ID, modelValue)
            setCustomKey(KEY_SESSION_ID, sessionValue)
            log("model=$modelValue session=$sessionValue")
        }
    }

    @JvmStatic
    fun reportLlmSetConfig(stage: String?, config: String?) {
        val safeStage = stage?.takeIf { it.isNotBlank() } ?: "unknown"
        val sourceConfig = config ?: ""
        val payload = buildLlmConfigLogPayload(
            stage = safeStage,
            config = sourceConfig,
            maxConfigChars = MAX_LLM_CONFIG_LOG_CHARS
        )
        updateSafely {
            setCustomKey(KEY_LLM_SET_CONFIG_STAGE, safeStage)
            setCustomKey(KEY_LLM_SET_CONFIG_SOURCE_LEN, sourceConfig.length)
            setCustomKey(KEY_LLM_SET_CONFIG_TRUNCATED, sourceConfig.length > MAX_LLM_CONFIG_LOG_CHARS)
            log(payload)
        }
    }

    internal fun buildLlmConfigLogPayload(stage: String?, config: String?, maxConfigChars: Int): String {
        val safeStage = stage?.takeIf { it.isNotBlank() } ?: "unknown"
        val normalizedConfig = config?.takeIf { it.isNotBlank() } ?: "{}"
        val safeMaxChars = if (maxConfigChars <= 0) 1 else maxConfigChars
        val maybeTruncatedConfig = if (normalizedConfig.length > safeMaxChars) {
            normalizedConfig.take(safeMaxChars) + "...(truncated, original_len=${normalizedConfig.length})"
        } else {
            normalizedConfig
        }
        return "[llm_set_config][$safeStage] len=${normalizedConfig.length} $maybeTruncatedConfig"
    }

    private inline fun updateSafely(block: FirebaseCrashlytics.() -> Unit) {
        try {
            FirebaseCrashlytics.getInstance().block()
        } catch (_: Throwable) {
            // Do not affect app behavior if crash reporting SDK is unavailable.
        }
    }
}
