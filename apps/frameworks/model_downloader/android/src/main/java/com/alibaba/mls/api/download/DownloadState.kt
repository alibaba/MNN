// Created by ruoyi.sjd on 2024/12/24.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mls.api.download

/**
 * Simple download state constants for the framework
 */
object DownloadState {
    const val NOT_START = 0
    const val DOWNLOADING = 1
    const val DOWNLOAD_SUCCESS = 2
    const val DOWNLOAD_FAILED = 3
    const val DOWNLOAD_PAUSED = 4
    const val DOWNLOAD_CANCELLED = 5
    const val PREPARING = 6

    // Aliases for compatibility
    const val COMPLETED = DOWNLOAD_SUCCESS
    const val FAILED = DOWNLOAD_FAILED
    const val PAUSED = DOWNLOAD_PAUSED
}
