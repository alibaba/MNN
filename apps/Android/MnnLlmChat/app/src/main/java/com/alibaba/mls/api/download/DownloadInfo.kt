// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mls.api.download

class DownloadInfo {
    @JvmField
    var totalSize: Long = 0
    @JvmField
    var savedSize: Long = 0
    @JvmField
    var progress: Double = 0.0

    @JvmField
    var lastLogTime: Long = 0

    @JvmField
    var downlodaState: Int = DownloadSate.NOT_START
    @JvmField
    var currentFile: String? = null
    @JvmField
    var progressStage: String? = null

    @JvmField
    var speedInfo: String? = null

    @JvmField
    var errorMessage: String? = null

    object DownloadSate {
        const val NOT_START: Int = 0

        const val DOWNLOADING: Int = 1

        const val COMPLETED: Int = 2

        const val FAILED: Int = 3

        const val PAUSED: Int = 4
    }
}

