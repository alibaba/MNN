// Created by ruoyi.sjd on 2024/12/24.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mls.api.download

/**
 * Utility functions for model ID manipulation
 */
object ModelIdUtils {

    /**
     * Convert modelId to safe format by replacing slashes with underscores
     * Example: "Huggingface/taobao-mnn/Qwen-1.5B" -> "Huggingface_taobao-mnn_Qwen-1.5B"
     */
    fun safeModelId(modelId: String): String {
        return modelId.replace("/".toRegex(), "_")
    }

    /**
     * Split "Huggingface/taobao-mnn/Qwen-1.5B" to ["Huggingface", "taobao-mnn/Qwen-1.5B"]
     */
    fun splitSource(modelId: String): Array<String> {
        val firstSlashIndex = modelId.indexOf('/')
        if (firstSlashIndex == -1) {
            return arrayOf(modelId)
        }
        val source = modelId.substring(0, firstSlashIndex)
        val repoPath = modelId.substring(firstSlashIndex + 1)
        return arrayOf(source, repoPath)
    }

    /**
     * Get repository path from model ID
     * Example: "Huggingface/taobao-mnn/Qwen-1.5B" -> "taobao-mnn/Qwen-1.5B"
     */
    fun getRepositoryPath(modelId: String): String {
        val parts = splitSource(modelId)
        return if (parts.size > 1) parts[1] else modelId
    }
}
