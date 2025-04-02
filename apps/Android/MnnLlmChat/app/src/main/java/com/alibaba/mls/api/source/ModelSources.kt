// Created by ruoyi.sjd on 2025/2/11.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mls.api.source

import com.alibaba.mls.api.ApplicationProvider
import com.alibaba.mnnllm.android.mainsettings.MainSettings.getDownloadProvider

class ModelSources {
    enum class ModelSourceType {
        MODEL_SCOPE,
        HUGGING_FACE,
        MODELERS,
        LOCAL
    }

    val remoteSourceType: ModelSourceType
        get() {
            return getDownloadProvider(ApplicationProvider.get())
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

        @JvmStatic
        fun get(): ModelSources {
            return InstanceHolder.instance
        }
    }
}
