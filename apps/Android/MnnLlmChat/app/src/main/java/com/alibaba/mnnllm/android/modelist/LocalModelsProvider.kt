package com.alibaba.mnnllm.android.modelist

import android.util.Log
import com.alibaba.mls.api.ModelItem
import org.json.JSONObject
import java.io.File

object LocalModelsProvider {
    /**
     * you can add ModelItem.fromLocalModel("Qwen-Omni-7B", "/data/local/tmp/omni_test/model")
     * to load local models
     */
    fun getLocalModels(): MutableList<ModelItem> {
        val result = mutableListOf<ModelItem>()
        try {
            val modelsDir = File("/data/local/tmp/mnn_models/")
            if (modelsDir.exists() && modelsDir.isDirectory) {
                modelsDir.listFiles()?.forEach { modelDir ->
                    if (modelDir.isDirectory && File(modelDir, "config.json").exists()) {
                        val modelPath = modelDir.absolutePath
                        val modelId = "local/${modelPath}"
                        val item = ModelItem.fromLocalModel(modelId, modelPath)
                        item.tags = readModalityTags(modelDir)
                        result.add(item)
                    }
                }
            }
        } catch (e: Exception) {
            Log.e("LocalModelsProvider", "Failed to load models from /data/local/tmp/mnn_models/", e)
        }
        return result
    }

    // Capability tags for a model placed on disk rather than downloaded. The market metadata
    // that normally supplies them only covers published models, so a local multimodal model is
    // otherwise offered no image / audio input at all, even though its own llm_config.json
    // already states what it accepts.
    private fun readModalityTags(modelDir: File): List<String> {
        val config = File(modelDir, "llm_config.json")
        if (!config.exists()) {
            return emptyList()
        }
        return try {
            val json = JSONObject(config.readText())
            val tags = mutableListOf<String>()
            if (json.optBoolean("is_visual", false)) {
                tags.add("Vision")
            }
            if (json.optBoolean("is_audio", false)) {
                tags.add("Audio")
            }
            tags
        } catch (e: Exception) {
            Log.w("LocalModelsProvider", "Failed to read ${config.absolutePath}", e)
            emptyList()
        }
    }
}
