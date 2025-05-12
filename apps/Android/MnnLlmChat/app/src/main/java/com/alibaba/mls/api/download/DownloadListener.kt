// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mls.api.download

interface DownloadListener {
    fun onDownloadTotalSize(modelId: String, totalSize:Long)
    fun onDownloadStart(modelId: String)
    fun onDownloadFailed(modelId: String, hfApiException: Exception)
    fun onDownloadProgress(modelId: String, progress: DownloadInfo)
    fun onDownloadFinished(modelId: String, path: String)
    fun onDownloadPaused(modelId: String)
    fun onDownloadFileRemoved(modelId: String)
}