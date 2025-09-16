// Created by ruoyi.sjd on 2025/7/17.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.modelmarket

import android.util.Log
import com.alibaba.mls.api.ApplicationProvider
import com.alibaba.mnnllm.android.modelsettings.ModelConfig
import com.google.gson.Gson
import com.google.gson.GsonBuilder
import com.google.gson.JsonParser
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

object ModelMarketUtils {

    const val TAG = "ModelMarketUtils"

    fun writeMarketConfig(modelItem: ModelMarketItem) {
        // Create a map with all fields except 'sources', and add 'modelId'
        val configMap = mutableMapOf<String, Any?>()
        configMap["modelName"] = modelItem.modelName
        configMap["vendor"] = modelItem.vendor
        configMap["size_gb"] = modelItem.sizeB
        configMap["tags"] = modelItem.tags
        configMap["extra_tags"] = modelItem.extraTags
        configMap["categories"] = modelItem.categories
        configMap["description"] = modelItem.description
        configMap["modelId"] = modelItem.modelId
        // Pretty-print JSON
        val gson = GsonBuilder().setPrettyPrinting().create()
        val jsonString = gson.toJson(configMap)
        // Write to file
        val filePath = ModelConfig.getMarketConfigFile(modelItem.modelId)
        try {
            val file = File(filePath)
            file.parentFile?.mkdirs()
            file.writeText(jsonString)
            Log.d(TAG, "Wrote market config to $filePath")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to write market config for ${modelItem.modelId}", e)
        }
    }


    suspend fun readMarketConfig(modelId:String):ModelMarketItem? {
        return withContext(Dispatchers.IO) {
            if (modelId.startsWith("Builtin/")) {
                Log.d(TAG, "Reading market config for builtin model: $modelId")
                return@withContext readMarketConfigFromLocal(modelId)
            }
            var marketItem: ModelMarketItem? = null
            try {
                val context = ApplicationProvider.get()
                val repository = ModelRepository(context)

                // Get current market data version from repository (network/cache/assets)
                val currentMarketVersion: String? = try {
                    repository.getModelMarketData()?.version
                } catch (e: Exception) {
                    Log.w(TAG, "Failed to get current market version from repository", e)
                    null
                }

                val localMarketVersion: String? = readLocalMarketVersion(modelId)
                val shouldRefreshFromRepository = (currentMarketVersion == null ||
                                                 localMarketVersion == null ||
                                                 localMarketVersion != currentMarketVersion)

                if (!shouldRefreshFromRepository) {
                    marketItem = readMarketConfigFromLocal(modelId)
                    if (marketItem != null) {
                        return@withContext marketItem
                    }
                } else {
                    Log.d(TAG, "Local market version ($localMarketVersion) != current market version ($currentMarketVersion), refreshing from repository")
                }
                try {
                    val context = ApplicationProvider.get()
                    val repository = ModelRepository(context)
                    // Get all models from different categories and find the one with matching modelId
                    val allModels = mutableListOf<ModelMarketItem>()
                    allModels.addAll(repository.getModels())
                    allModels.addAll(repository.getTtsModels()) 
                    allModels.addAll(repository.getAsrModels())
                    marketItem = allModels.find { it.modelId == modelId }
                    if (marketItem != null) {
                        writeMarketConfig(marketItem)
                    }
                } catch (e: Exception) {
                    Log.w(TAG, "Failed to read from ModelRepository for $modelId", e)
                }
            }
            marketItem
        }
    }

    suspend fun readMarketConfigFromLocal(modelId:String):ModelMarketItem? {
        return withContext(Dispatchers.IO) {
            val marketConfigFile = ModelConfig.getMarketConfigFile(modelId)
            Log.d(TAG, "readMarketConfigFromLocal for $modelId: looking for file at $marketConfigFile")
            var marketItem: ModelMarketItem? = null
            try {
                val configFile = File(marketConfigFile)
                Log.d(TAG, "Config file exists: ${configFile.exists()}")
                if (configFile.exists()) {
                    val configJson = configFile.readText()
                    Log.d(TAG, "Config file content: $configJson")
                    marketItem = Gson().fromJson(configJson, ModelMarketItem::class.java)
                    Log.d(TAG, "Loaded market config for ${modelId}: ${marketItem.modelName}, tags=${marketItem.tags}")
                } else {
                    Log.w(TAG, "Market config file does not exist: $marketConfigFile")
                }
            } catch (e: Exception) {
                Log.w(TAG, "Failed to load market config for ${modelId}", e)
            }
            marketItem
        }
    }
}