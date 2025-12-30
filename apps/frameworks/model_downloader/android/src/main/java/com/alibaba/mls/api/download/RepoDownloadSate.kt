// Created by ruoyi.sjd on 2024/12/24.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mls.api.download

/**
 * Repository download state - simplified for framework
 */
data class RepoDownloadSate(
    var modelId: String? = null,
    var state: Int = DownloadState.NOT_START,
    var downlodaState: Int = DownloadState.NOT_START,  // Typo preserved for compatibility
    var progress: Double = 0.0,
    var totalSize: Long = 0,
    var savedSize: Long = 0,
    var errorMessage: String? = null
) {
    companion object {
        const val COMPLETED = DownloadState.DOWNLOAD_SUCCESS
        const val FAILED = DownloadState.DOWNLOAD_FAILED
        const val PAUSED = DownloadState.DOWNLOAD_PAUSED
        const val DOWNLOADING = DownloadState.DOWNLOADING
        const val NOT_START = DownloadState.NOT_START
    }
}
