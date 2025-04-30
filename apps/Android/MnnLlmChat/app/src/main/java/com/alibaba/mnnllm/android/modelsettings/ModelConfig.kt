// Created by ruoyi.sjd on 2025/4/29.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.modelsettings

import android.util.Log
import com.alibaba.mnnllm.android.utils.FileUtils
import com.google.gson.Gson
import com.google.gson.GsonBuilder
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import java.io.File
import com.google.gson.annotations.SerializedName

data class ModelConfig(
    @SerializedName("llm_model") var llmModel: String?,
    @SerializedName("llm_weight") var llmWeight: String?,
    @SerializedName("backend_type") var backendType: String?,
    @SerializedName("thread_num") var threadNum: Int?,
    @SerializedName("precision") var precision: String?,
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
    @SerializedName("assistant_prompt_template")var assistantPromptTemplate:String?
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
            assistantPromptTemplate = this.assistantPromptTemplate
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
                this.nGramFactor == loadedConfig.nGramFactor
    }

    companion object {

        const val TAG = "ModelConfig"

        fun loadConfig(filePath: String): ModelConfig? {
            return try {
                val file = File(filePath)
                val json = file.readText()
                Gson().fromJson(json, ModelConfig::class.java)
            } catch (e: Exception) {
                e.printStackTrace()
                null
            }
        }

        fun loadConfig(originalFilePath: String, overrideFilePath: String): ModelConfig? {
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

        private fun mergeJson(original: JsonObject, override: JsonObject) {
            for (key in override.keySet()) {
                original.add(key, override.get(key))
            }
        }

        fun toJson(): String {
            return Gson().toJson(this)
        }

        fun saveConfig(filePath: String, config: ModelConfig): Boolean {
            return try {
                Log.d(TAG, "file is : $filePath")
                val file = File(filePath)
                FileUtils.ensureParentDirectoriesExist(file)
                val gson = GsonBuilder().setPrettyPrinting().create()
                val jsonString = gson.toJson(config)
                file.writeText(jsonString)
                true
            } catch (e: Exception) {
                Log.e(TAG, "saveConfig error", e)
                false
            }
        }

        fun saveConfigOld(filePath: String, config: ModelConfig): Boolean {
            return try {
                val file = File(filePath)
                FileUtils.ensureParentDirectoriesExist(file)
                val jsonObject = JsonObject()

                if (config.llmModel != null) jsonObject.addProperty(
                    "llm_model",
                    config.llmModel
                )
                if (config.llmWeight != null) jsonObject.addProperty(
                    "llm_weight",
                    config.llmWeight
                )
                if (config.backendType != null) jsonObject.addProperty(
                    "backend_type",
                    config.backendType
                )
                if (config.maxNewTokens != null) jsonObject.addProperty("max_new_tokens", config.maxNewTokens)
                if (config.threadNum != null) jsonObject.addProperty("threadNum", config.threadNum)
                if (config.nGram != null) jsonObject.addProperty("n_gram", config.nGram)
                if (config.precision!= null) jsonObject.addProperty(
                    "precision",
                    config.precision
                )
                if (config.memory!= null) jsonObject.addProperty("memory", config.memory)
                if (config.systemPrompt!= null) jsonObject.addProperty(
                    "system_prompt",
                    config.systemPrompt
                )
                if (config.samplerType != null) jsonObject.addProperty(
                    "sampler_type",
                    config.samplerType
                )
                if (config.mixedSamplers != null && config.mixedSamplers!!.isNotEmpty()) jsonObject.add(
                    "mixed_samplers",
                    Gson().toJsonTree(config.mixedSamplers)
                )
                if (config.temperature != null) jsonObject.addProperty(
                    "temperature",
                    config.temperature
                )
                if (config.tfsZ != null) jsonObject.addProperty(
                    "tfsZ",
                    config.tfsZ
                )
                if (config.typical != null) jsonObject.addProperty(
                    "typical",
                    config.typical
                )
                if (config.penalty != null) jsonObject.addProperty(
                    "penalty",
                    config.penalty
                )
                if (config.nGramFactor != null) jsonObject.addProperty(
                    "ngram_factor",
                    config.nGramFactor
                )
                if (config.topP != null) jsonObject.addProperty("topP", config.topP)
                if (config.topK != null) jsonObject.addProperty("topK", config.topK)
                if (config.minP != null) jsonObject.addProperty("minP", config.minP)

                file.writeText(Gson().toJson(jsonObject))
                true
            } catch (e: Exception) {
                e.printStackTrace()
                false
            }
        }
    }
}

