// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited. All rights reserved.
package com.alibaba.mls.api

import com.alibaba.mnnllm.android.model.ModelUtils.generateSimpleTags
import com.alibaba.mnnllm.android.model.ModelUtils.getModelName

class ModelItem {
    var modelId: String? = null
    var createdAt: String? = null
    var downloads: Int = 0
    var localPath:String? = null
    private val tags: MutableList<String> = mutableListOf()

    private var cachedTags: List<String>? = null

    val modelName: String?
        get() = getModelName(modelId)

    val isLocal: Boolean
        get() = !localPath.isNullOrEmpty()

    fun getTags(): List<String> = tags

    val newTags: List<String>
        get() {
            if (cachedTags == null) {
                cachedTags = modelName?.let { generateSimpleTags(it, this) } ?: emptyList()
            }
            return cachedTags ?: emptyList()
        }

    fun addTag(tag: String) {
        tags.add(tag)
    }

    companion object {
        fun fromLocalModel(modelId:String, path:String):ModelItem {
            return ModelItem().apply {
                this.modelId = modelId
                this.localPath = path
            }
        }
    }
}