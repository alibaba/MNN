// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mls.api.download

data class DownloadInfo(
    var downloadState: Int = DownloadState.NOT_START,
    var progress: Double = 0.0,
    var savedSize: Long = 0,
    var totalSize: Long = 0,
    var speedInfo: String = "",
    var errorMessage: String? = null,
    var errorException: Exception? = null,
    var lastLogTime: Long = 0,
    var lastProgressUpdateTime: Long = 0,
    var progressStage: String = "",
    var currentFile: String? = null,
    var downloadedTime: Long = 0L,
    var remoteUpdateTime: Long = 0L,
    var hasUpdate: Boolean = false
)

