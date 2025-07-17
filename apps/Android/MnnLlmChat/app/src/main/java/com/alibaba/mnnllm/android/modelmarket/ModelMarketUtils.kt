// Created by ruoyi.sjd on 2025/7/17.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.modelmarket

import android.util.Log
import com.alibaba.mnnllm.android.modelsettings.ModelConfig
import com.google.gson.GsonBuilder
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

}