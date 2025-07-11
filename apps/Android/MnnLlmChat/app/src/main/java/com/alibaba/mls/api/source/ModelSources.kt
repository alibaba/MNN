// Created by ruoyi.sjd on 2025/2/11.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mls.api.source

import com.alibaba.mls.api.ApplicationProvider
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.mainsettings.MainSettings.getDownloadProvider

class ModelSources {

    enum class ModelSourceType {
        MODEL_SCOPE,
        HUGGING_FACE,
        MODELERS,
    }

    val remoteSourceType: ModelSourceType
        get() {
            return getDownloadProvider(ApplicationProvider.get())
        }

    private object InstanceHolder {
        val instance: ModelSources = ModelSources()
    }

    companion object {

        @JvmStatic
        fun get(): ModelSources {
            return InstanceHolder.instance
        }

        const val sourceHuffingFace = "HuggingFace"
        const val sourceModelScope = "ModelScope"
        const val sourceModelers = "Modelers"
        val sourceList = listOf(sourceHuffingFace, sourceModelScope, sourceModelers)
        val sourceDisPlayList = listOf(R.string.huggingface, R.string.modelscope, R.string.modelers)
    }
}
