// Created by ruoyi.sjd on 2025/4/29.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.modelsettings

import android.util.Log
import com.alibaba.mls.api.ApplicationProvider
import com.alibaba.mls.api.download.ModelDownloadManager
import com.alibaba.mnnllm.android.model.ModelUtils
import com.alibaba.mnnllm.android.utils.FileUtils
import com.google.gson.Gson
import com.google.gson.GsonBuilder
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import java.io.File
import com.google.gson.annotations.SerializedName

data class JinjaContext(
    @SerializedName("enable_thinking") var enableThinking: Boolean = false
)

data class Jinja(
    @SerializedName("context") var context: JinjaContext? = null
)

data class ModelConfig(
    @SerializedName("llm_model") var llmModel: String?,
    @SerializedName("llm_weight") var llmWeight: String?,
    @SerializedName("backend_type") var backendType: String?,
    @SerializedName("thread_num") var threadNum: Int?,
    @SerializedName("precision") var precision: String?,
    @SerializedName("use_mmap") var useMmap: Boolean?,
    @SerializedName("memory") var memory: String?,
    @SerializedName("system_prompt") var systemPrompt: String?,
    @SerializedName("sampler_type") var samplerType: String?,
    @SerializedName("mixed_samplers") var mixedSamplers: MutableList<String>?,
    @SerializedName("temperature") var temperature: Float?,
    @SerializedName("topP") var topP: Float?,
    @SerializedName("topK") var topK: Int?,
    @SerializedName("minP") var minP: Float?,
    var tfsZ:Float?,
    var typical:Float?,
    var penalty:Float?,
    @SerializedName("n_gram")var nGram:Int?,
    @SerializedName("ngram_factor")var nGramFactor:Float?,
    @SerializedName("max_new_tokens")var maxNewTokens:Int?,
    @SerializedName("assistant_prompt_template")var assistantPromptTemplate:String?,
    @SerializedName("penalty_sampler")var penaltySampler:String?,
    @SerializedName("jinja") var jinja: Jinja?
    ) {
    fun deepCopy(): ModelConfig {
        return ModelConfig(
            llmModel = this.llmModel,
            llmWeight = this.llmWeight,
            backendType = this.backendType,
            threadNum = this.threadNum,
            precision = this.precision,
            memory = this.memory,
            systemPrompt = this.systemPrompt,
            samplerType = this.samplerType,
            mixedSamplers = this.mixedSamplers?.toMutableList(),
            temperature = this.temperature,
            topP = this.topP,
            topK = this.topK,
            minP = this.minP,
            tfsZ = this.tfsZ,
            typical = this.typical,
            penalty = this.penalty,
            nGram = this.nGram,
            nGramFactor = this.nGramFactor,
            maxNewTokens = this.maxNewTokens,
            assistantPromptTemplate = this.assistantPromptTemplate,
            penaltySampler = this.penaltySampler,
            useMmap =  this.useMmap,
            jinja = this.jinja?.let { 
                Jinja(context = JinjaContext(enableThinking = it.context?.enableThinking == true))
            }
        )
    }

    fun samplerEquals(loadedConfig: ModelConfig): Boolean {
        return this.samplerType == loadedConfig.samplerType &&
                this.mixedSamplers == loadedConfig.mixedSamplers &&
                this.temperature == loadedConfig.temperature &&
                this.topP == loadedConfig.topP &&
                this.topK == loadedConfig.topK &&
                this.minP == loadedConfig.minP &&
                this.tfsZ == loadedConfig.tfsZ &&
                this.typical == loadedConfig.typical &&
                this.penalty == loadedConfig.penalty &&
                this.nGram == loadedConfig.nGram &&
                this.nGramFactor == loadedConfig.nGramFactor &&
                this.penaltySampler == loadedConfig.penaltySampler
    }

    companion object {

        const val TAG = "ModelConfig"

        fun loadDefaultConfig(filePath: String): ModelConfig? {
            return try {
                val file = File(filePath)
                val json = file.readText()
                Gson().fromJson(json, ModelConfig::class.java)
            } catch (e: Exception) {
                e.printStackTrace()
                null
            }
        }

        fun loadConfig(modelId: String): ModelConfig? {
            return loadMergedConfig(getDefaultConfigFile(modelId)!!, getExtraConfigFile(modelId))
        }

        fun loadMergedConfig(originalFilePath: String, overrideFilePath: String): ModelConfig? {
            return try {
                val originalFile = File(originalFilePath)
                val originalJson = JsonParser.parseString(originalFile.readText()).asJsonObject

                val overrideFile = File(overrideFilePath)
                if (overrideFile.exists()) {
                    val overrideJson = JsonParser.parseString(overrideFile.readText()).asJsonObject
                    mergeJson(originalJson, overrideJson)
                }
                Gson().fromJson(originalJson, ModelConfig::class.java)
            } catch (e: Exception) {
                e.printStackTrace()
                null
            }
        }

        fun getDefaultConfigFile(modelId:String):String? {
            if (modelId.startsWith("local/")) {
                val localPath = modelId.removePrefix("local/")
                val configFilePath = File(localPath, "config.json")
                if (configFilePath.exists()) {
                    return configFilePath.absolutePath
                }
                return null
            }
            val configFileName = "config.json"
            val destModelDir = ModelDownloadManager.getInstance(ApplicationProvider.get())
                .getDownloadedFile(modelId)?.absolutePath
            destModelDir?.let {
                val configFilePath = File(destModelDir, configFileName)
                if (configFilePath.exists()) {
                    return configFilePath.absolutePath
                }
            }
            return null
        }

        private fun mergeJson(original: JsonObject, override: JsonObject) {
            for (key in override.keySet()) {
                original.add(key, override.get(key))
            }
        }

        fun toJson(): String {
            return GsonBuilder()
                .disableHtmlEscaping()
                .create()
                .toJson(this)
        }

        fun saveConfig(filePath: String, config: ModelConfig): Boolean {
            return try {
                Log.d(TAG, "file is : $filePath")
                val file = File(filePath)
                FileUtils.ensureParentDirectoriesExist(file)
                val gson = GsonBuilder()
                    .setPrettyPrinting()
                    .disableHtmlEscaping()
                    .create()
                val jsonString = gson.toJson(config)
                file.writeText(jsonString)
                true
            } catch (e: Exception) {
                Log.e(TAG, "saveConfig error", e)
                false
            }
        }

        fun getExtraConfigFile(modelId: String):String {
            return getModelConfigDir(modelId) + "/custom_config.json"
        }

        fun getMarketConfigFile(modelId: String):String {
            return getModelConfigDir(modelId) + "/market_config.json"
        }

        fun getModelConfigDir(modelId: String): String {
            val rootCacheDir =
                ApplicationProvider.get().filesDir.toString() + "/configs/" + ModelUtils.safeModelId(
                    modelId
                )
            return rootCacheDir
        }

        val defaultConfig:ModelConfig = ModelConfig (
            llmModel = "",
            llmWeight = "",
            backendType = "",
            threadNum = 4,
            precision = "low",
            memory = "",
            systemPrompt = "You are a helpful assistant.",
            samplerType = "",
            mixedSamplers = mutableListOf("topK", "topP", "minP", "temperature"),
            temperature = 0.6f,
            topP = 0.95f,
            topK = 20,
            minP = 0.05f,
            tfsZ = 1.0f,
            typical = 0.95f,
            penalty = 1.02f,
            nGram = 8,
            nGramFactor = 1.02f,
            maxNewTokens = 2048,
            assistantPromptTemplate = "",
            penaltySampler = "greedy",
            useMmap = false,
            jinja = null
        )

    }
}

