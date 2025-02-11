// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited. All rights reserved.
package com.alibaba.mls.api;

import java.util.ArrayList;
import java.util.List;

public class HfRepoInfo {
    public static class SiblingItem {
        public String rfilename;
    }

    private String modelId;
    private String revision;
    private String sha;
    private List<SiblingItem> siblings;

    public HfRepoInfo() {
        siblings = new ArrayList<>();
    }

    // Getters and Setters
    public String getModelId() { return modelId; }
    public void setModelId(String modelId) { this.modelId = modelId; }

    public String getRevision() { return revision; }
    public void setRevision(String revision) { this.revision = revision; }

    public String getSha() { return sha; }
    public void setSha(String sha) { this.sha = sha; }

    public List<SiblingItem> getSiblings() { return siblings; }
    public void addSibling(SiblingItem sibling) { this.siblings.add(sibling); }
}
