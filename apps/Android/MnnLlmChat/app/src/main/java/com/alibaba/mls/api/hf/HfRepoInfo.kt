// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited. All rights reserved.
package com.alibaba.mls.api.hf

class HfRepoInfo {
    class SiblingItem {
        @JvmField
        var rfilename: String? = null
    }

    // Getters and Setters
    @JvmField
    var modelId: String? = null
    var revision: String? = null
    @JvmField
    var sha: String? = null
    private val siblings: MutableList<SiblingItem> =
        ArrayList()

    fun getSiblings(): List<SiblingItem> {
        return siblings
    }

    fun addSibling(sibling: SiblingItem) {
        siblings.add(sibling)
    }
}
