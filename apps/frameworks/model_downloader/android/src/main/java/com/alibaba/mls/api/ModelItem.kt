// Created by ruoyi.sjd on 2024/12/24.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mls.api

/**
 * Simple model item for framework - apps can extend this
 */
data class ModelItem(
    var modelId: String? = null,
    var modelName: String? = null,
    var localPath: String? = null,
    @get:JvmName("_getTags")  // Rename auto-generated getter to avoid clash
    var tags: List<String> = emptyList(),
    var vendor: String? = null,
    var isBuiltin: Boolean = false,
    var sizeB: Long = 0,
    var modelMarketItem: Any? = null  // For compatibility - apps can store their ModelMarketItem here
) {
    val isLocal: Boolean
        get() = !localPath.isNullOrEmpty()

    // Explicit getTags() method for compatibility
    fun getTags(): List<String> {
        return tags
    }

    fun getExtraTags(): List<String> {
        return emptyList()  // Apps can override
    }

    fun getDisplayTags(): List<String> {
        return tags
    }

    fun getDisplayTags(context: android.content.Context): List<String> {
        return getDisplayTags()
    }

    companion object {
        fun fromLocalModel(modelId: String, path: String): ModelItem {
            return ModelItem(modelId = modelId, localPath = path)
        }

        fun fromDownloadModel(context: android.content.Context, modelId: String, path: String): ModelItem {
            // Stub - apps can override with more sophisticated logic
            return ModelItem(modelId = modelId, localPath = path)
        }
    }
}
