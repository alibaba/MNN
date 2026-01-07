// Created by ruoyi.sjd on 2025/2/11.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mls.api.source

class ModelSourceConfig {
    var repos: LinkedHashMap<String, RepoConfig>? = null

    fun getRepoConfig(modelId: String): RepoConfig? {
        if (repos == null) {
            return null
        }
        var repoConfig = repos!![modelId]
        if (repoConfig == null) {
            val splits = modelId.split("/".toRegex()).dropLastWhile { it.isEmpty() }.toTypedArray()
            if (splits.size == 2) {
                val modelScopeId = "MNN/" + splits[1]
                repoConfig = RepoConfig(modelScopeId, modelId, modelId)
            } else {
                val modelScopeId = "MNN/$modelId"
                val huggingFaceId = "taobao-mnn/$modelId"
                repoConfig = RepoConfig(modelScopeId, huggingFaceId, huggingFaceId)
            }
            repos!![modelId] = repoConfig
        }
        return repoConfig
    }

    companion object {
        //we might create a server to store the model source configs, but now use some hack
        // to download both modelscope and huggingface models
        fun createMockSourceConfig(): ModelSourceConfig {
            val config = ModelSourceConfig()
            config.repos = LinkedHashMap()
            return config
        }
    }
}
