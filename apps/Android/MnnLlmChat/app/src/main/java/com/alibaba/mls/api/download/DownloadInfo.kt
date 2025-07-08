// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mls.api.download

data class DownloadInfo(
    @JvmField
    var totalSize: Long = 0,
    @JvmField
    var savedSize: Long = 0,
    @JvmField
    var progress: Double = 0.0,
    @JvmField
    var lastLogTime: Long = 0,
    @JvmField
    var downloadState: Int = DownloadState.NOT_START,
    @JvmField
    var currentFile: String? = null,
    @JvmField
    var progressStage: String? = null,
    @JvmField
    var speedInfo: String? = null,
    @JvmField
    var errorMessage: String? = null
)

