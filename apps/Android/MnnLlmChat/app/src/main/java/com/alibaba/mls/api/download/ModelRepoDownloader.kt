// Created by ruoyi.sjd on 2025/5/8.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mls.api.download

import com.alibaba.mls.api.download.DownloadFileUtils.getLastFileName
import java.io.File
import java.util.Collections

abstract class ModelRepoDownloader {
    abstract var callback: ModelRepoDownloadCallback?
    abstract var cacheRootPath:String
    protected val pausedSet: MutableSet<String?> = Collections.synchronizedSet(HashSet())
    abstract fun setListener(callback: ModelRepoDownloadCallback?)
    abstract fun download(modelId: String)
    fun pause(modelId: String) {
        pausedSet.add(modelId)
    }
    fun getDownloadedFile(modelId: String): File {
        return File(cacheRootPath, getLastFileName(modelId))
    }
    abstract fun resume(modelId: String)
    abstract fun cancel(modelId: String)
    abstract fun getDownloadPath(modelId: String): File
    abstract fun isDownloaded(modelId: String): Boolean
    abstract fun deleteRepo(modelId: String)

    interface ModelRepoDownloadCallback {
        fun onDownloadFailed(modelId: String, e: Exception)
        fun onDownloadTaskAdded()
        fun onDownloadTaskRemoved()
        fun onDownloadPaused(modelId: String)
        fun onDownloadFileFinished(modelId: String, absolutePath: String)
        fun onDownloadingProgress(
            modelId: String,
            stage: String,
            currentFile: String?,
            saved: Long,
            total: Long)
    }
}