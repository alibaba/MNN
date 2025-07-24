// Created by ruoyi.sjd on 2025/5/8.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mls.api.download.ms
import android.util.Log
import com.alibaba.mls.api.ApplicationProvider
import com.alibaba.mls.api.FileDownloadException
import com.alibaba.mls.api.hf.HfFileMetadata
import com.alibaba.mls.api.download.DownloadExecutor.Companion.executor
import com.alibaba.mls.api.download.DownloadFileUtils.createSymlink
import com.alibaba.mls.api.download.DownloadFileUtils.deleteDirectoryRecursively
import com.alibaba.mls.api.download.DownloadFileUtils.getLastFileName
import com.alibaba.mls.api.download.DownloadFileUtils.getPointerPathParent
import com.alibaba.mls.api.download.DownloadFileUtils.repoFolderName
import com.alibaba.mls.api.download.DownloadPausedException
import com.alibaba.mls.api.download.FileDownloadTask
import com.alibaba.mls.api.download.ModelDownloadManager.Companion.TAG
import com.alibaba.mls.api.download.ModelFileDownloader
import com.alibaba.mls.api.download.ModelFileDownloader.FileDownloadListener
import com.alibaba.mls.api.download.ModelRepoDownloader
import com.alibaba.mls.api.ms.MsApiClient
import com.alibaba.mls.api.ms.MsRepoInfo
import com.alibaba.mnnllm.android.chat.model.ChatDataManager
import com.alibaba.mnnllm.android.model.ModelUtils
import com.alibaba.mnnllm.android.utils.TimeUtils
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import retrofit2.Call
import retrofit2.Callback
import retrofit2.Response
import java.io.File

class MsModelDownloader(override var callback: ModelRepoDownloadCallback?,
                        cacheRootPath: String
) : ModelRepoDownloader() {
    override var cacheRootPath: String = getCachePathRoot(cacheRootPath)
    private var msApiClient: MsApiClient = MsApiClient()
    override fun setListener(callback: ModelRepoDownloadCallback?) {
        this.callback = callback
    }

    /**
     * Unified method to fetch repo information from ModelScope API
     * @param modelId the model ID to fetch info for
     * @param calculateSize whether to calculate repo size (not needed for ModelScope as size is included)
     * @return MsRepoInfo object or null if failed
     */
    private suspend fun fetchRepoInfo(modelId: String, calculateSize: Boolean = false): MsRepoInfo? {
        return withContext(Dispatchers.IO) {
            runCatching {
                val msModelId = ModelUtils.getRepositoryPath(modelId)
                val split = msModelId.split("/".toRegex()).dropLastWhile { it.isEmpty() }.toTypedArray()
                if (split.size != 2) {
                    return@runCatching null
                }
                
                val response = msApiClient.apiService.getModelFiles(split[0], split[1]).execute()
                if (response.isSuccessful && response.body() != null) {
                    val repoInfo = response.body()!!
                    
                    // Call onRepoInfo callback with repo metadata
                    val lastModified = repoInfo.Data?.LatestCommitter?.CreatedAt ?: 0L
                    val repoSize = repoInfo.Data?.Files?.filter { it.Type != "tree" }?.sumOf { it.Size } ?: 0L
                    callback?.onRepoInfo(modelId, lastModified, repoSize)
                    repoInfo
                } else {
                    null
                }
            }.getOrElse { exception ->
                Log.e(TAG, "Failed to fetch repo info for $modelId", exception)
                null
            }
        }
    }

    override fun download(modelId: String) {
        executor!!.submit {
            kotlinx.coroutines.runBlocking {
                downloadMsRepo(modelId)
            }
        }
    }

    override suspend fun checkUpdate(modelId: String) {
        fetchRepoInfo(modelId)
    }

    override fun getDownloadPath(modelId: String): File {
        return getModelPath(cacheRootPath, modelId)
    }

    override fun deleteRepo(modelId: String) {
        val msModelId = ModelUtils.getRepositoryPath(modelId)
        val msRepoFolderName = repoFolderName(msModelId, "model")
        val msStorageFolder = File(this.cacheRootPath, msRepoFolderName)
        Log.d(TAG, "removeStorageFolder: " + msStorageFolder.absolutePath)
        if (msStorageFolder.exists()) {
            val result = deleteDirectoryRecursively(msStorageFolder)
            if (!result) {
                Log.e(TAG, "remove storageFolder" + msStorageFolder.absolutePath + " faield")
            }
        }
        val msLinkFolder = this.getDownloadPath(modelId)
        Log.d(TAG, "removeMsLinkFolder: " + msLinkFolder.absolutePath)
        msLinkFolder.delete()
    }

    override suspend fun getRepoSize(modelId: String): Long {
        return withContext(Dispatchers.IO) {
            runCatching {
                val repoInfo = fetchRepoInfo(modelId, calculateSize = true)
                if (repoInfo != null) {
                    repoInfo.Data?.Files?.filter { it.Type != "tree" }?.sumOf { it.Size } ?: 0L
                } else {
                    0L
                }
            }.getOrElse { exception ->
                Log.e(TAG, "Failed to get repo size for $modelId", exception)
                0L
            }
        }
    }

    private suspend fun downloadMsRepo(modelId: String) {
        val modelScopeId = ModelUtils.getRepositoryPath(modelId)
        val split = modelScopeId.split("/".toRegex()).dropLastWhile { it.isEmpty() }.toTypedArray()
        if (split.size != 2) {
            callback?.onDownloadFailed(modelId, FileDownloadException("getRepoInfoFailed modelId format error: $modelId"))
            return
        }
        
        val repoInfo = fetchRepoInfo(modelId, calculateSize = true)
        if (repoInfo != null) {
            Log.d(TAG, "downloadMsRepoInner executor")
            callback?.onDownloadTaskAdded()
            downloadMsRepoInner(modelId, modelScopeId, repoInfo)
            callback?.onDownloadTaskRemoved()
        } else {
            callback?.onDownloadFailed(modelId, FileDownloadException("Failed to get repo info for $modelId"))
        }
    }

    private fun downloadMsRepoInner(modelId:String, modelScopeId: String, msRepoInfo: MsRepoInfo) {
        Log.d(TAG, "downloadMsRepoInner")
        val folderLinkFile =
            File(cacheRootPath, getLastFileName(modelScopeId))
        if (folderLinkFile.exists()) {
            Log.d(TAG, "downloadMsRepoInner already exists")
            callback?.onDownloadFileFinished(modelId, folderLinkFile.absolutePath)
            return
        }
        val modelDownloader = ModelFileDownloader()
        val hasError = false
        val errorInfo = StringBuilder()
        val repoFolderName = repoFolderName(modelScopeId, "model")
        val storageFolder = File(this.cacheRootPath, repoFolderName)
        val parentPointerPath = getPointerPathParent(storageFolder, "_no_sha_")
        val downloadTaskList: List<FileDownloadTask>
        val totalAndDownloadSize = LongArray(2)
        Log.d(TAG, "downloadMsRepoInner collectMsTaskList")
        downloadTaskList = collectMsTaskList(
            modelScopeId,
            storageFolder,
            parentPointerPath,
            msRepoInfo,
            totalAndDownloadSize
        )
        Log.d(TAG, "downloadMsRepoInner downloadTaskList： " + downloadTaskList.size)
        val fileDownloadListener =
            object : FileDownloadListener {
                override fun onDownloadDelta(
                    fileName: String?,
                    downloadedBytes: Long,
                    totalBytes: Long,
                    delta: Long
                ): Boolean {
                    totalAndDownloadSize[1] += delta
                    callback?.onDownloadingProgress(
                        modelId, "file", fileName,
                        totalAndDownloadSize[1], totalAndDownloadSize[0]
                    )
                    return pausedSet.contains(modelId)
                }
            }
        try {
            for (fileDownloadTask in downloadTaskList) {
                modelDownloader.downloadFile(fileDownloadTask, fileDownloadListener)
            }
        } catch (e: DownloadPausedException) {
            pausedSet.remove(modelId)
            callback?.onDownloadPaused(modelId)
            return
        } catch (e: Exception) {
            callback?.onDownloadFailed(modelId, e)
            return
        }
        if (!hasError) {
            val folderLinkPath = folderLinkFile.absolutePath
            createSymlink(parentPointerPath.toString(), folderLinkPath)
            callback?.onDownloadFileFinished(modelId, folderLinkPath)
        } else {
            Log.e(
                TAG,
                "Errors occurred during download: $errorInfo"
            )
        }
    }

    private fun collectMsTaskList(
        modelScopeId: String,
        storageFolder: File,
        parentPointerPath: File,
        msRepoInfo: MsRepoInfo,
        totalAndDownloadSize: LongArray
    ): List<FileDownloadTask> {
        val fileDownloadTasks: MutableList<FileDownloadTask> = ArrayList()
        for (i in msRepoInfo.Data?.Files?.indices!!) {
            val subFile = msRepoInfo.Data!!.Files!![i]
            val fileDownloadTask = FileDownloadTask()
            if (subFile.Type == "tree") {
                continue
            }
            fileDownloadTask.relativePath = subFile.Path
            fileDownloadTask.fileMetadata = HfFileMetadata()
            fileDownloadTask.fileMetadata!!.location = String.format(
                "https://modelscope.cn/api/v1/models/%s/repo?FilePath=%s",
                modelScopeId,
                subFile.Path
            )
            fileDownloadTask.fileMetadata!!.size = subFile.Size
            fileDownloadTask.fileMetadata!!.etag = subFile.Sha256
            fileDownloadTask.etag = subFile.Sha256
            fileDownloadTask.blobPath = File(storageFolder, "blobs/" + subFile.Sha256)
            fileDownloadTask.blobPathIncomplete =
                File(storageFolder, "blobs/" + subFile.Sha256 + ".incomplete")
            fileDownloadTask.pointerPath = File(parentPointerPath, subFile.Path)
            fileDownloadTask.downloadedSize =
                if (fileDownloadTask.blobPath!!.exists()) fileDownloadTask.blobPath!!.length() else (if (fileDownloadTask.blobPathIncomplete!!.exists()) fileDownloadTask.blobPathIncomplete!!.length() else 0)
            totalAndDownloadSize[0] += subFile.Size
            totalAndDownloadSize[1] += fileDownloadTask.downloadedSize
            fileDownloadTasks.add(fileDownloadTask)
        }
        return fileDownloadTasks
    }


    companion object  {
        fun getCachePathRoot(modelDownloadPathRoot:String): String {
            return "$modelDownloadPathRoot/modelscope"
        }

        fun getModelPath(modelsDownloadPathRoot: String, modelId: String): File {
            return File(modelsDownloadPathRoot, getLastFileName(modelId))
        }
    }

}
