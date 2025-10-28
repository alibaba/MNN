// Created by ruoyi.sjd on 2025/7/18.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.modelist

import com.alibaba.mls.api.ModelItem
import com.alibaba.mnnllm.android.chat.model.ChatDataManager
import com.alibaba.mnnllm.android.utils.FileUtils

/**
 * Wrapper class that combines ModelItem with DownloadedModelInfo and download size
 */
data class ModelItemWrapper(
    val modelItem: ModelItem,
    val downloadedModelInfo: ChatDataManager.DownloadedModelInfo? = null,
    val downloadSize: Long = 0,
    var isPinned: Boolean = false,
    var hasUpdate: Boolean = false
) {
    val displayName: String
        get() = modelItem.modelName ?: ""

    val isLocal: Boolean
        get() = modelItem.isLocal

    val lastChatTime: Long
        get() = downloadedModelInfo?.lastChatTime ?: 0

    val downloadTime: Long
        get() = downloadedModelInfo?.downloadTime ?: 0

    val formattedSize: String
        get() = if (downloadSize > 0) FileUtils.formatFileSize(downloadSize) else ""
}