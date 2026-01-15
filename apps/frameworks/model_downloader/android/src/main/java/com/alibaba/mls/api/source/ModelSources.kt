// Created by ruoyi.sjd on 2025/2/11.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mls.api.source

import com.alibaba.mls.api.download.R

class ModelSources {
    enum class ModelSourceType {
        MODEL_SCOPE,
        HUGGING_FACE,
        MODELERS,
        LOCAL
    }

    val remoteSourceType: ModelSourceType
        get() {
            return ModelSourceType.MODEL_SCOPE
        }

    private object InstanceHolder {
        val instance: ModelSources = ModelSources()
    }

    val config: ModelSourceConfig
        get() {
            if (mockConfig == null) {
                mockConfig = ModelSourceConfig.createMockSourceConfig()
            }
            return mockConfig!!
        }

    companion object {
        private var mockConfig: ModelSourceConfig? = null

        const val sourceHuffingFace = "HuggingFace"
        const val sourceModelers = "Modelers"
        const val sourceModelScope = "ModelScope"

        val sourceList = listOf(
            sourceHuffingFace,
            sourceModelers,
            sourceModelScope
        )

        val sourceDisPlayList = listOf(
            R.string.source_huggingface,
            R.string.source_modelers,
            R.string.source_modelscope
        )

        @JvmStatic
        fun get(): ModelSources {
            return InstanceHolder.instance
        }
    }
}
