// Created by ruoyi.sjd on 2025/2/11.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mls.api.source
import com.alibaba.mls.api.source.ModelSources.Companion.get

class RepoConfig(var modelScopePath: String,
                 private var huggingFacePath: String,
                 var modelId: String) {
    fun repositoryPath(): String {
        return if (get().remoteSourceType == ModelSources.ModelSourceType.HUGGING_FACE)
            huggingFacePath
            else
            modelScopePath
    }
}