// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited. All rights reserved.

package com.alibaba.mls.api;
import com.alibaba.mnnllm.android.utils.ModelUtils;

import java.util.ArrayList;
import java.util.List;

public class HfRepoItem {
    private String modelId;
    private String createdAt;
    private int downloads;
    private List<String> tags;

    private List<String> new_tags;

    public HfRepoItem() {
        tags = new ArrayList<>();
    }

    // Getters and Setters
    public String getModelId() { return modelId; }
    public void setModelId(String modelId) { this.modelId = modelId; }

    public String getCreatedAt() { return createdAt; }

    public String getModelName() {
        if (modelId != null && modelId.contains("/")) {
            return modelId.substring(modelId.lastIndexOf("/") + 1);
        }
        return modelId;
    }
    public void setCreatedAt(String createdAt) { this.createdAt = createdAt; }

    public int getDownloads() { return downloads; }
    public void setDownloads(int downloads) { this.downloads = downloads; }

    public List<String> getTags() { return tags; }

    public List<String> getNewTags() {
        if (new_tags == null) {
            new_tags = ModelUtils.generateSimpleTags(getModelName());
        }
        return new_tags;
    }
    public void addTag(String tag) { this.tags.add(tag); }
}
