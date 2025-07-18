// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited. All rights reserved.
package com.alibaba.mls.api.hf

import com.alibaba.mls.api.source.ModelSources

class HfRepoInfo {
    class SiblingItem {
        var rfilename: String? = null
    }

    var modelId: String? = null
    var revision: String? = null
    var sha: String? = null
    var lastModified:String?= null
    private val siblings: MutableList<SiblingItem> =
        ArrayList()

    fun getSiblings(): List<SiblingItem> {
        return siblings
    }

    fun addSibling(sibling: SiblingItem) {
        siblings.add(sibling)
    }

    fun universalModelId(): String {
        return "${ModelSources.sourceHuffingFace}/$modelId"
    }
}
