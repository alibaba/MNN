// Created by ruoyi.sjd on 2025/6/20.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.utils

import com.alibaba.mls.api.ApplicationProvider
import com.alibaba.mls.api.download.DownloadFileUtils
import com.alibaba.mls.api.source.ModelSources
import com.alibaba.mnnllm.android.model.ModelUtils
import java.io.File

object MmapUtils {

    fun clearMmapCache(modelId: String):Boolean {
        return DownloadFileUtils.deleteDirectoryRecursively(File(getMmapDir(modelId)))
    }

    fun getMmapDir(modelId: String): String {
        var newModelId = modelId
        val isModelScope = modelId.startsWith(ModelSources.sourceModelScope)
        val isModelers = modelId.startsWith(ModelSources.sourceModelers)
        val isHuggingFace = modelId.startsWith(ModelSources.sourceHuffingFace)
        if (isModelers || isHuggingFace || isModelScope) {
            newModelId = modelId.substring(modelId.indexOf("/") + 1)
        }
        if (newModelId.startsWith("MNN/")) {
            newModelId = newModelId.replace("MNN/", "taobao-mnn/")
        }
        var rootCacheDir =
            ApplicationProvider.get().filesDir.toString() + "/tmps/" + ModelUtils.safeModelId(
                newModelId
            )
        if (isModelScope) {
            rootCacheDir = "$rootCacheDir/modelscope"
        } else if (isModelers) {
            rootCacheDir = "$rootCacheDir/modelers"
        }
        return rootCacheDir
    }
}