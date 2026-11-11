package com.alibaba.mnnllm.android.model

import android.util.Log
import org.json.JSONObject
import java.io.File
import java.util.concurrent.ConcurrentHashMap

/**
 * Input capabilities a model declares in its exported llm_config.json. A null flag means
 * the field is absent, which callers must not read as "unsupported": older packages omit
 * it and the name/tag heuristics still apply to them.
 */
data class ModelCapabilities(val isVisual: Boolean?, val isAudio: Boolean?) {
    companion object {
        val UNKNOWN = ModelCapabilities(null, null)
    }
}

object ModelCapabilityReader {
    private const val TAG = "ModelCapabilityReader"
    private const val LLM_CONFIG = "llm_config.json"

    private data class CacheKey(val path: String, val lastModified: Long, val length: Long)

    // Read on UI paths (input bar, attachment picker), so parse once per file version.
    // The key includes mtime and size, so a re-downloaded model is read again.
    private val cache = ConcurrentHashMap<CacheKey, ModelCapabilities>()

    fun read(modelDir: File): ModelCapabilities {
        val config = File(modelDir, LLM_CONFIG)
        if (!config.isFile) {
            return ModelCapabilities.UNKNOWN
        }
        val key = CacheKey(config.absolutePath, config.lastModified(), config.length())
        cache[key]?.let { return it }
        val parsed = parse(config)
        cache.keys.removeAll { it.path == key.path }
        cache[key] = parsed
        return parsed
    }

    /** [configPath] is the model's config.json; llm_config.json sits next to it. */
    fun readForConfigPath(configPath: String?): ModelCapabilities {
        if (configPath.isNullOrEmpty()) {
            return ModelCapabilities.UNKNOWN
        }
        val dir = File(configPath).let { if (it.isDirectory) it else it.parentFile } ?: return ModelCapabilities.UNKNOWN
        return read(dir)
    }

    fun modalityTags(modelDir: File): List<String> {
        val capabilities = read(modelDir)
        return buildList {
            if (capabilities.isVisual == true) add("Vision")
            if (capabilities.isAudio == true) add("Audio")
        }
    }

    private fun parse(config: File): ModelCapabilities {
        return try {
            val json = JSONObject(config.readText())
            ModelCapabilities(booleanOrNull(json, "is_visual"), booleanOrNull(json, "is_audio"))
        } catch (e: Exception) {
            Log.w(TAG, "Failed to read ${config.absolutePath}", e)
            ModelCapabilities.UNKNOWN
        }
    }

    private fun booleanOrNull(json: JSONObject, key: String): Boolean? {
        if (!json.has(key) || json.isNull(key)) {
            return null
        }
        return json.opt(key) as? Boolean
    }
}
