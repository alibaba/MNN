// Created by ruoyi.sjd on 2025/4/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.taobao.meta.avatar.download

import java.io.File

interface DownloadCallback {
    fun onDownloadStart()
    fun onDownloadProgress(progress: Double,
                           currentBytes: Long,
                           totalBytes: Long,
                           speedInfo:String)
    fun onDownloadComplete(success: Boolean, file: File?)
    fun onDownloadError(error: Exception?)
}