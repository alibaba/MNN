// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited. All rights reserved.
package com.alibaba.mls.api

import android.content.Context
import com.alibaba.mls.api.source.ModelSources
import com.alibaba.mnnllm.android.model.ModelUtils
import com.alibaba.mnnllm.android.model.ModelUtils.getModelName
import com.alibaba.mnnllm.android.utils.DeviceUtils
import com.alibaba.mnnllm.android.modelmarket.ModelMarketItem
import com.alibaba.mnnllm.android.modelmarket.ModelRepository

class ModelItem {
    var modelId: String? = null
    var localPath:String? = null
    private val tags: MutableList<String> = mutableListOf()
    private var source:String? = null
    private var marketTags: List<String>? = null // Cache for tags from model_market.json
    var modelMarketItem: ModelMarketItem? = null // Market item data from market_config.json

    val modelName: String
        get() {
            val name = modelMarketItem?.modelName ?: getModelName(modelId) ?: modelId
            if (name == null) {
                throw IllegalStateException("ModelItem.modelName is null! modelId=$modelId, modelMarketItem=${modelMarketItem?.modelName}")
            }
            return name
        }

    val isLocal: Boolean
        get() = !localPath.isNullOrEmpty()

    val isBuiltin: Boolean
        get() = modelId?.startsWith("Builtin/") == true

    fun getTags(): List<String> {
        return when {
            modelMarketItem != null -> modelMarketItem!!.tags
            !marketTags.isNullOrEmpty() -> marketTags!!
            else -> emptyList()
        }
    }

    fun getExtraTags(): List<String> {
        return modelMarketItem?.extraTags ?: emptyList()
    }

    fun addTag(tag: String) {
        tags.add(tag)
    }

    fun getDisplayTags(context: Context): List<String> {
        val tags = mutableListOf<String>()

        // Add source tag first
        val source = modelId?.let { ModelUtils.getModelSource(context, modelId) }
        if (source != null) {
            tags.add(source)
        }

        // Use getTags() which now prioritizes market tags from model_market.json
        val marketTags = getTags()

        // Add builtin/local/downloaded status
        if (isBuiltin) {
            tags.add(context.getString(com.alibaba.mnnllm.android.R.string.builtin))
        } else if (isLocal) {
            tags.add(context.getString(com.alibaba.mnnllm.android.R.string.local))
        }
        
        // Add market tags for all models (including builtin and local)
        if (marketTags.isNotEmpty()) {
            // Use ModelRepository to translate market tags to Chinese only in Chinese mode
            val tagsToAdd = if (DeviceUtils.isChinese) {
                ModelRepository.getTagTranslations(marketTags.take(2))
            } else {
                marketTags.take(2)
            }
            tags.addAll(tagsToAdd)
        }

        // Limit total tags to 3 for better UI layout
        return tags.take(3)
    }

    companion object {
        fun fromLocalModel(modelId:String, path:String):ModelItem {
            return ModelItem().apply {
                this.modelId = modelId
                this.localPath = path
            }
        }

        fun fromDownloadModel(context: Context, modelId:String, path:String):ModelItem {
            return ModelItem().apply {
                this.modelId = modelId
                this.source = ModelUtils.getSource(modelId)!!
            }
        }
    }
}