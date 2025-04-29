// Created by ruoyi.sjd on 2025/4/29.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.modelsettings

import com.google.gson.Gson
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import java.io.File

data class ModelConfig(
    var llmModel: String,
    var llmWeight: String,
    var backendType: String,
    var threadNum: Int,
    var precision: String,
    var memory: String,
    var systemPrompt: String,
    var samplerType: String,
    var mixedSamplers: List<String>,
    var temperature: Double,
    var topP: Double,
    var topK: Int,
    var minP: Double
) {
    companion object {

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

        fun saveConfig(filePath: String, config: ModelConfig): Boolean {
            return try {
                val file = File(filePath)
                val jsonObject = JsonObject()

                // Add only non-default values to the JSON object
                if (config.llmModel.isNotEmpty()) jsonObject.addProperty(
                    "llmModel",
                    config.llmModel
                )
                if (config.llmWeight.isNotEmpty()) jsonObject.addProperty(
                    "llmWeight",
                    config.llmWeight
                )
                if (config.backendType.isNotEmpty()) jsonObject.addProperty(
                    "backendType",
                    config.backendType
                )
                if (config.threadNum != 0) jsonObject.addProperty("threadNum", config.threadNum)
                if (config.precision.isNotEmpty()) jsonObject.addProperty(
                    "precision",
                    config.precision
                )
                if (config.memory.isNotEmpty()) jsonObject.addProperty("memory", config.memory)
                if (config.systemPrompt.isNotEmpty()) jsonObject.addProperty(
                    "systemPrompt",
                    config.systemPrompt
                )
                if (config.samplerType.isNotEmpty()) jsonObject.addProperty(
                    "samplerType",
                    config.samplerType
                )
                if (config.mixedSamplers.isNotEmpty()) jsonObject.add(
                    "mixedSamplers",
                    Gson().toJsonTree(config.mixedSamplers)
                )
                if (config.temperature != 0.0) jsonObject.addProperty(
                    "temperature",
                    config.temperature
                )
                if (config.topP != 0.0) jsonObject.addProperty("topP", config.topP)
                if (config.topK != 0) jsonObject.addProperty("topK", config.topK)
                if (config.minP != 0.0) jsonObject.addProperty("minP", config.minP)

                file.writeText(Gson().toJson(jsonObject))
                true
            } catch (e: Exception) {
                e.printStackTrace()
                false
            }
        }
    }
}

