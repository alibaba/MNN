// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mls.api.download

class DownloadInfo {
    var totalSize: Long = 0
    var savedSize: Long = 0
    var progress: Double = 0.0

    //for calculate speed
    var lastLogTime: Long = 0
    var downlodaState: Int = RepoDownloadSate.NOT_START
    var currentFile: String? = null
    var progressStage: String? = null
    var speedInfo: String? = null
    var errorMessage: String? = null

    fun isDownloadingOrPreparing():Boolean {
        return downlodaState == RepoDownloadSate.DOWNLOADING
                || downlodaState == RepoDownloadSate.PREPARING
    }

    fun isDownloading():Boolean {
        return downlodaState == RepoDownloadSate.DOWNLOADING
    }

    fun canDownload():Boolean {
        return listOf(
            RepoDownloadSate.NOT_START,
            RepoDownloadSate.PAUSED,
            RepoDownloadSate.FAILED
        ).contains(downlodaState)
    }

    fun isComplete(): Boolean {
        return downlodaState == RepoDownloadSate.COMPLETED
    }
}

