package com.alibaba.mnnllm.android.llm

import com.alibaba.mnnllm.android.modelsettings.ModelConfig
import com.google.gson.Gson

internal object DiffusionLoadConfigResolver {
    fun buildExtraConfigJson(modelId: String, configPath: String): String {
        val config = ModelConfig.loadMergedConfig(configPath, ModelConfig.getExtraConfigFile(modelId))
            ?: ModelConfig.loadDefaultConfig(configPath)
            ?: ModelConfig.defaultConfig
        val configMap = hashMapOf<String, Any>(
            "diffusion_memory_mode" to (config.diffusionMemoryMode
                ?: ModelConfig.defaultConfig.diffusionMemoryMode
                ?: "0")
        )
        return Gson().toJson(configMap)
    }
}
