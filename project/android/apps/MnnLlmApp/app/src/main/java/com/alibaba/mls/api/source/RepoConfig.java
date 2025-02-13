// Created by ruoyi.sjd on 2025/2/11.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mls.api.source;

public class RepoConfig {
    public String modelScopePath;
    public String huggingFacePath;
    public String modelId;

    public RepoConfig(String repositoryPath, String huggingFaceId, String modelId) {
        this.modelScopePath = repositoryPath;
        this.huggingFacePath = huggingFaceId;
        this.modelId = modelId;
    }

    public String repositoryPath() {
        return ModelSources.get().getRemoteSourceType() == ModelSources.ModelSourceType.HUGGING_FACE ? huggingFacePath : modelScopePath;
    }
}