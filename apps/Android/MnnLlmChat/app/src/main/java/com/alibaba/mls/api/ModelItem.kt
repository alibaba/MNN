// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited. All rights reserved.
package com.alibaba.mls.api

import android.content.Context
import com.alibaba.mnnllm.android.model.ModelUtils
import com.alibaba.mnnllm.android.model.ModelUtils.getModelName
import com.alibaba.mnnllm.android.modelmarket.SourceSelectionDialogFragment
import com.alibaba.mnnllm.android.modelmarket.TagMapper

class ModelItem {
    var modelId: String? = null
    var localPath:String? = null
    private val tags: MutableList<String> = mutableListOf()
    private var source:String? = null
    private var marketTags: List<String>? = null // Cache for tags from model_market.json

    val modelName: String?
        get() = getModelName(modelId)

    val isLocal: Boolean
        get() = !localPath.isNullOrEmpty()

    fun getTags(): List<String> {
        return when {
            !marketTags.isNullOrEmpty() -> marketTags!!
            else -> emptyList()
        }
    }

    fun addTag(tag: String) {
        tags.add(tag)
    }

    fun getDisplayTags(): List<String> {
        return TagMapper.getDisplayTagList(getTags())
    }

    /**
     * Load market tags from cache
     */
    fun loadMarketTags(context: Context) {
        if (modelId != null) {
            marketTags = ModelTagsCache.getTagsForModel(context, modelId!!)
        }
    }

    /**
     * Set market tags directly (for testing or special cases)
     */
    fun setMarketTags(tags: List<String>) {
        marketTags = tags
    }

    companion object {
        fun fromLocalModel(modelId:String, path:String):ModelItem {
            return ModelItem().apply {
                this.modelId = modelId
                this.localPath = path
            }
        }

        fun fromDownloadModel(modelId:String, path:String):ModelItem {
            return ModelItem().apply {
                this.modelId = modelId
                this.addTag(sourceToTag(ModelUtils.getSource(modelId)!!))
            }
        }

        // New factory methods with Context for proper tag loading
        fun fromLocalModel(context: Context, modelId:String, path:String):ModelItem {
            return ModelItem().apply {
                this.modelId = modelId
                this.localPath = path
                this.loadMarketTags(context)
            }
        }

        fun fromDownloadModel(context: Context, modelId:String, path:String):ModelItem {
            return ModelItem().apply {
                this.modelId = modelId
                this.source = ModelUtils.getSource(modelId)!!
                this.addTag(sourceToTag(ModelUtils.getSource(modelId)!!))
                this.loadMarketTags(context)
            }
        }

        fun sourceToTag(source:String):String {
            return when (source) {
                SourceSelectionDialogFragment.SOURCE_HUGGINGFACE -> "hf"
                SourceSelectionDialogFragment.SOURCE_MODELERS -> "ml"
                else -> "ms"
            }
        }
    }
}