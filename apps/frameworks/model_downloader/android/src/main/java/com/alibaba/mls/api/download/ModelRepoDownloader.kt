// Created by ruoyi.sjd on 2025/5/8.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mls.api.download

import com.alibaba.mls.api.download.DownloadFileUtils.repoFolderName
import com.alibaba.mls.api.download.ModelIdUtils
import java.io.File
import java.util.Collections

abstract class ModelRepoDownloader {
    abstract var callback: ModelRepoDownloadCallback?
    abstract var cacheRootPath:String
    protected val pausedSet: MutableSet<String?> = Collections.synchronizedSet(HashSet())
    abstract suspend fun getRepoSize(modelId: String):Long
    abstract fun setListener(callback: ModelRepoDownloadCallback?)
    abstract fun download(modelId: String)

    fun repoModelRealFile(modelId: String):File {
        return File(cacheRootPath, "${repoFolderName(ModelIdUtils.getRepositoryPath(modelId), "model")}/blobs")
    }

    abstract suspend fun checkUpdate(modelId: String)

    fun pause(modelId: String) {
        pausedSet.add(modelId)
    }

    fun isDownloaded(modelId: String): Boolean {
        return getDownloadPath(modelId).exists()
    }

    abstract fun getDownloadPath(modelId: String): File
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
        fun onRepoInfo(modelId: String, lastModified: Long, repoSize: Long)
    }
}