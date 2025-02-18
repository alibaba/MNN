// Created by ruoyi.sjd on 2025/2/11.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mls.api.source;

import java.util.LinkedHashMap;

public class ModelSourceConfig {

    public LinkedHashMap<String, RepoConfig> repos;

    //we might create a server to store the model source configs, but now use some hack
    // to download both modelscope and huggingface models
    public static ModelSourceConfig createMockSourceConfig() {
        ModelSourceConfig config = new ModelSourceConfig();
        config.repos = new LinkedHashMap<>();
        return config;
    }

    public RepoConfig getRepoConfig(String modelId) {
        if (repos == null) {
            return null;
        }
        RepoConfig repoConfig = repos.get(modelId);
        if (repoConfig == null) {
            String[] splits = modelId.split("/");
            if (splits.length == 2) {
                String modelScopeId = "MNN/" + splits[1];
                repoConfig = new RepoConfig(modelScopeId, modelId, modelId);
            } else {
                String modelScopeId = "MNN/" + modelId;
                String huggingFaceId = "taobao-mnn/" + modelId;
                repoConfig = new RepoConfig(modelScopeId, huggingFaceId, huggingFaceId);
            }
            repos.put(modelId, repoConfig);
        }
        return repoConfig;
    }
}
