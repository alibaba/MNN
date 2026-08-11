package com.alibaba.mnnllm.android.utils

import com.google.gson.JsonParser
import com.google.firebase.crashlytics.FirebaseCrashlytics
import kotlin.concurrent.Volatile

/**
 * Centralized Crashlytics context updater so crash reports always include
 * latest UI and model execution state.
 */
object CrashReportContext {
    private const val KEY_CURRENT_ACTIVITY = "current_activity"
    private const val KEY_APP_IN_FOREGROUND = "app_in_foreground"
    private const val KEY_MODEL_ID = "model_id"
    private const val KEY_SESSION_ID = "session_id"
    private const val KEY_BENCHMARK_STATE = "benchmark_state"
    private const val KEY_BENCHMARK_MODEL_ID = "benchmark_model_id"
    private const val KEY_BENCHMARK_BACKEND = "benchmark_backend"
    private const val KEY_BENCHMARK_N_PROMPT = "benchmark_n_prompt"
    private const val KEY_BENCHMARK_N_GENERATE = "benchmark_n_generate"
    private const val KEY_BENCHMARK_N_REPEAT = "benchmark_n_repeat"
    private const val KEY_BENCHMARK_KV_CACHE = "benchmark_kv_cache"
    private const val KEY_LLM_SET_CONFIG_STAGE = "llm_set_config_stage"
    private const val KEY_LLM_SET_CONFIG_SOURCE_LEN = "llm_set_config_source_len"
    private const val KEY_LLM_SET_CONFIG_TRUNCATED = "llm_set_config_truncated"
    private const val KEY_LLM_SET_CONFIG_USE_MMAP = "use_mmap"
    private const val KEY_LLM_SET_CONFIG_BACKEND = "backend"
    private const val MAX_LLM_CONFIG_LOG_CHARS = 3000
    private val NON_LLM_CONFIG_KEYS = setOf(
        "diffusion_memory_mode",
        "diffusion_steps",
        "image_width",
        "image_height",
        "diffusion_seed",
        "cfg_prompt",
        "grid_size"
    )
    @Volatile
    private var lastKnownActivityName: String = "none"

    fun setCurrentActivity(activityName: String?) {
        val value = resolveCurrentActivityForCrashReport(activityName, lastKnownActivityName)
        if (!activityName.isNullOrBlank()) {
            lastKnownActivityName = value
        }
        updateSafely {
            setCustomKey(KEY_CURRENT_ACTIVITY, value)
            log("activity=$value")
        }
    }

    fun setAppInForeground(inForeground: Boolean) {
        updateSafely {
            setCustomKey(KEY_APP_IN_FOREGROUND, inForeground)
            log("app_in_foreground=$inForeground")
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

    fun setBenchmarkState(state: String?) {
        val safeState = state?.takeIf { it.isNotBlank() } ?: "unknown"
        updateSafely {
            setCustomKey(KEY_BENCHMARK_STATE, safeState)
            log("benchmark_state=$safeState")
        }
    }

    fun setBenchmarkContext(
        modelId: String?,
        backend: String?,
        nPrompt: Int,
        nGenerate: Int,
        nRepeat: Int,
        kvCache: String?
    ) {
        val payload = buildBenchmarkContextLogPayload(
            modelId = modelId,
            backend = backend,
            nPrompt = nPrompt,
            nGenerate = nGenerate,
            nRepeat = nRepeat,
            kvCache = kvCache
        )
        val safeModelId = modelId?.takeIf { it.isNotBlank() } ?: "none"
        val safeBackend = backend?.takeIf { it.isNotBlank() } ?: "cpu"
        val safePrompt = nPrompt.coerceAtLeast(0)
        val safeGenerate = nGenerate.coerceAtLeast(0)
        val safeRepeat = nRepeat.coerceAtLeast(1)
        val safeKvCache = (kvCache ?: "false").equals("true", ignoreCase = true)
        updateSafely {
            setCustomKey(KEY_BENCHMARK_MODEL_ID, safeModelId)
            setCustomKey(KEY_BENCHMARK_BACKEND, safeBackend)
            setCustomKey(KEY_BENCHMARK_N_PROMPT, safePrompt)
            setCustomKey(KEY_BENCHMARK_N_GENERATE, safeGenerate)
            setCustomKey(KEY_BENCHMARK_N_REPEAT, safeRepeat)
            setCustomKey(KEY_BENCHMARK_KV_CACHE, safeKvCache)
            log(payload)
        }
    }

    @JvmStatic
    fun reportLlmSetConfig(stage: String?, config: String?) {
        val safeStage = stage?.takeIf { it.isNotBlank() } ?: "unknown"
        val sourceConfig = sanitizeLlmConfigForLog(config)
        val crashKeys = extractLlmConfigCrashKeys(sourceConfig)
        val payload = buildLlmConfigLogPayload(
            stage = safeStage,
            config = sourceConfig,
            maxConfigChars = MAX_LLM_CONFIG_LOG_CHARS
        )
        updateSafely {
            setCustomKey(KEY_LLM_SET_CONFIG_STAGE, safeStage)
            setCustomKey(KEY_LLM_SET_CONFIG_SOURCE_LEN, sourceConfig.length)
            setCustomKey(KEY_LLM_SET_CONFIG_TRUNCATED, sourceConfig.length > MAX_LLM_CONFIG_LOG_CHARS)
            setCustomKey(KEY_LLM_SET_CONFIG_USE_MMAP, crashKeys.useMmap)
            setCustomKey(KEY_LLM_SET_CONFIG_BACKEND, crashKeys.backend)
            log(payload)
        }
    }

    internal fun resolveCurrentActivityForCrashReport(activityName: String?, lastKnownActivity: String?): String {
        val current = activityName?.takeIf { it.isNotBlank() }
        if (current != null) {
            return current
        }
        return lastKnownActivity?.takeIf { it.isNotBlank() } ?: "none"
    }

    internal fun buildBenchmarkContextLogPayload(
        modelId: String?,
        backend: String?,
        nPrompt: Int,
        nGenerate: Int,
        nRepeat: Int,
        kvCache: String?
    ): String {
        val safeModelId = modelId?.takeIf { it.isNotBlank() } ?: "none"
        val safeBackend = backend?.takeIf { it.isNotBlank() } ?: "cpu"
        val safePrompt = nPrompt.coerceAtLeast(0)
        val safeGenerate = nGenerate.coerceAtLeast(0)
        val safeRepeat = nRepeat.coerceAtLeast(1)
        val safeKvCache = (kvCache ?: "false").equals("true", ignoreCase = true)
        return "[benchmark] model=$safeModelId backend=$safeBackend prompt=$safePrompt generate=$safeGenerate repeat=$safeRepeat kv_cache=$safeKvCache"
    }

    internal fun buildLlmConfigLogPayload(stage: String?, config: String?, maxConfigChars: Int): String {
        val safeStage = stage?.takeIf { it.isNotBlank() } ?: "unknown"
        val normalizedConfig = sanitizeLlmConfigForLog(config)
        val safeMaxChars = if (maxConfigChars <= 0) 1 else maxConfigChars
        val maybeTruncatedConfig = if (normalizedConfig.length > safeMaxChars) {
            normalizedConfig.take(safeMaxChars) + "...(truncated, original_len=${normalizedConfig.length})"
        } else {
            normalizedConfig
        }
        return "[llm_set_config][$safeStage] len=${normalizedConfig.length} $maybeTruncatedConfig"
    }

    internal fun sanitizeLlmConfigForLog(config: String?): String {
        val normalizedConfig = config?.takeIf { it.isNotBlank() } ?: "{}"
        val parsed = runCatching { JsonParser.parseString(normalizedConfig) }.getOrNull() ?: return normalizedConfig
        if (!parsed.isJsonObject) {
            return normalizedConfig
        }
        val jsonObject = parsed.asJsonObject
        NON_LLM_CONFIG_KEYS.forEach { key ->
            jsonObject.remove(key)
        }
        return jsonObject.toString()
    }

    internal fun extractLlmConfigCrashKeys(config: String?): LlmConfigCrashKeys {
        val normalizedConfig = config?.takeIf { it.isNotBlank() } ?: "{}"
        val parsed = runCatching { JsonParser.parseString(normalizedConfig) }.getOrNull()
            ?: return LlmConfigCrashKeys(backend = "unknown", useMmap = false)
        if (!parsed.isJsonObject) {
            return LlmConfigCrashKeys(backend = "unknown", useMmap = false)
        }

        val jsonObject = parsed.asJsonObject
        val backend = parseStringKey(jsonObject, "backend_type")
            ?: parseStringKey(jsonObject, "backend")
            ?: "unknown"
        val useMmap = parseBooleanKey(jsonObject, "use_mmap")
            ?: parseBooleanKey(jsonObject, "use_mma")
            ?: false
        return LlmConfigCrashKeys(backend = backend, useMmap = useMmap)
    }

    private fun parseStringKey(jsonObject: com.google.gson.JsonObject, key: String): String? {
        val element = jsonObject.get(key) ?: return null
        if (!element.isJsonPrimitive) {
            return null
        }
        val primitive = element.asJsonPrimitive
        return when {
            primitive.isString -> primitive.asString.takeIf { it.isNotBlank() }
            primitive.isNumber -> primitive.asNumber.toString()
            primitive.isBoolean -> primitive.asBoolean.toString()
            else -> null
        }
    }

    private fun parseBooleanKey(jsonObject: com.google.gson.JsonObject, key: String): Boolean? {
        val element = jsonObject.get(key) ?: return null
        if (!element.isJsonPrimitive) {
            return null
        }
        val primitive = element.asJsonPrimitive
        return when {
            primitive.isBoolean -> primitive.asBoolean
            primitive.isNumber -> primitive.asInt != 0
            primitive.isString -> when (primitive.asString.trim().lowercase()) {
                "true", "1", "yes", "on" -> true
                "false", "0", "no", "off" -> false
                else -> null
            }
            else -> null
        }
    }

    internal data class LlmConfigCrashKeys(
        val backend: String,
        val useMmap: Boolean
    )

    private inline fun updateSafely(block: FirebaseCrashlytics.() -> Unit) {
        try {
            FirebaseCrashlytics.getInstance().block()
        } catch (_: Throwable) {
            // Do not affect app behavior if crash reporting SDK is unavailable.
        }
    }
}
