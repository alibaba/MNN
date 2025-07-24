// Created by ruoyi.sjd on 2025/5/8.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mls.api.download.ml
import android.util.Log
import com.alibaba.mls.api.ApplicationProvider
import com.alibaba.mls.api.FileDownloadException
import com.alibaba.mls.api.download.DownloadExecutor
import com.alibaba.mls.api.hf.HfFileMetadata
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
import com.alibaba.mls.api.ml.FileInfo
import com.alibaba.mls.api.ml.MlApiClient
import com.alibaba.mls.api.ml.MlRepoInfo
import com.alibaba.mnnllm.android.model.ModelUtils
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import com.alibaba.mls.api.ml.MlRepoData
import com.alibaba.mnnllm.android.utils.TimeUtils


class MLModelDownloader(override var callback: ModelRepoDownloadCallback?,
                        cacheRootPath: String
) : ModelRepoDownloader() {
    override var cacheRootPath: String = getCachePathRoot(cacheRootPath)
    private var mlApiClient: MlApiClient = MlApiClient()
    override fun setListener(callback: ModelRepoDownloadCallback?) {
        this.callback = callback
    }

    /**
     * Unified method to fetch repo information from Modelers API
     * @param modelId the model ID to fetch info for
     * @param calculateSize whether to calculate repo size (requires additional network requests)
     * @return MlRepoInfo object or null if failed
     */
    private suspend fun fetchRepoInfo(modelId: String, calculateSize: Boolean = false): MlRepoInfo? {
        return withContext(Dispatchers.IO) {
            runCatching {
                val modelersId = ModelUtils.getRepositoryPath(modelId)
                val split = modelersId.split("/".toRegex()).dropLastWhile { it.isEmpty() }.toTypedArray()
                if (split.size != 2) {
                    return@runCatching null
                }
                
                val response = mlApiClient.apiService.getModelFiles(split[0], split[1], "").execute()
                if (response.isSuccessful && response.body() != null) {
                    val initialInfo = response.body()!!
                    val repoInfo = if (calculateSize) {
                        // Get all files recursively for size calculation
                        val allFiles = mutableListOf<FileInfo>()
                        getAllFiles(split[0], split[1], "", allFiles)
                        MlRepoInfo(
                            initialInfo.code,
                            initialInfo.msg,
                            MlRepoData(
                                tree = allFiles,
                                last_commit = initialInfo.data.last_commit,
                                commit_count = initialInfo.data.commit_count
                            )
                        )
                    } else {
                        initialInfo
                    }
                    
                    // Call onRepoInfo callback with repo metadata
                    val lastModified = TimeUtils.convertIsoToTimestamp(repoInfo.data.last_commit?.commit?.created) ?: 0L
                    val repoSize = if (calculateSize && repoInfo.data.tree.isNotEmpty()) {
                        repoInfo.data.tree.filter { it.type != "dir" }.sumOf { it.size }
                    } else {
                        0L
                    }
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
        Log.d(TAG, "start download  ${modelId}")
        DownloadExecutor.executeScope.launch {
            downloadRepo(modelId)
        }
    }

    override suspend fun checkUpdate(modelId: String) {
        fetchRepoInfo(modelId, false)
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
                    repoInfo.data.tree.filter { it.type != "dir" }.sumOf { it.size }
                } else {
                    0L
                }
            }.getOrElse { exception ->
                Log.e(TAG, "Failed to get repo size for $modelId", exception)
                0L
            }
        }
    }

    private suspend fun downloadRepo(modelId: String): MlRepoInfo? {
        val modelersId = ModelUtils.getRepositoryPath(modelId)
        Log.d(TAG, "downloadRepo: $modelersId")
        val split = modelersId.split("/".toRegex()).dropLastWhile { it.isEmpty() }.toTypedArray()
        if (split.size != 2) {
            callback?.onDownloadFailed(modelId, FileDownloadException("getRepoInfoFailed modelId format error: $modelId"))
            return null
        }
        
        val modelInfo = fetchRepoInfo(modelId, calculateSize = true)
        if (modelInfo != null) {
            callback?.onDownloadTaskAdded()
            downloadMlRepoInner(modelId, modelersId, modelInfo)
            callback?.onDownloadTaskRemoved()
        } else {
            callback?.onDownloadFailed(modelId, FileDownloadException("Failed to get repo info for $modelId"))
        }
        return modelInfo
    }

    private fun getAllFiles(owner: String, repo: String, path: String, allFiles: MutableList<FileInfo>) {
        Log.d(TAG, "getAllFiles: owner: $owner path $path repo:$repo")
        kotlin.runCatching {
            val response = mlApiClient.apiService.getModelFiles(owner, repo, path).execute()
            val repoInfo = response.body()
            
            repoInfo?.data?.tree?.forEach { fileInfo ->
                allFiles.add(fileInfo)
                // If it's a directory, recursively get its files
                if (fileInfo.type == "dir") {
                    getAllFiles(owner, repo, fileInfo.path, allFiles)
                }
            }
        }.getOrElse { error ->
            Log.e(TAG, "Failed to get files for path $path: ${error.message}")
        }
    }
    private fun downloadMlRepoInner(modelId:String, modelScopeId: String, mlRepoInfo: MlRepoInfo) {
        Log.d(TAG, "downloadMlRepoInner")
        val folderLinkFile =
            File(cacheRootPath, getLastFileName(modelScopeId))
        if (folderLinkFile.exists()) {
            Log.d(TAG, "downloadMlRepoInner already exists")
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
        Log.d(TAG, "downloadMlRepoInner collectMsTaskList")
        downloadTaskList = collectTaskList(
            modelScopeId,
            storageFolder,
            parentPointerPath,
            mlRepoInfo,
            totalAndDownloadSize
        )
        Log.d(TAG, "downloadMlRepoInner downloadTaskList： " + downloadTaskList.size)
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

    private fun collectTaskList(
        modelScopeId: String,
        storageFolder: File,
        parentPointerPath: File,
        mlRepoInfo: MlRepoInfo,
        totalAndDownloadSize: LongArray
    ): List<FileDownloadTask> {
        val fileDownloadTasks: MutableList<FileDownloadTask> = ArrayList()
        for (i in mlRepoInfo.data.tree.indices) {
            val subFile = mlRepoInfo.data.tree[i]
            if (subFile.type == "dir") {
                continue
            }
            val fileDownloadTask = FileDownloadTask()
            fileDownloadTask.relativePath = subFile.path
            fileDownloadTask.fileMetadata = HfFileMetadata()
            fileDownloadTask.fileMetadata!!.location = String.format(
                "https://modelers.cn/coderepo/web/v1/file/%s/main/media/%s",
                modelScopeId,
                subFile.path
            )
            fileDownloadTask.fileMetadata!!.size = subFile.size
            fileDownloadTask.fileMetadata!!.etag = subFile.etag
            fileDownloadTask.etag = subFile.etag
            fileDownloadTask.blobPath = File(storageFolder, "blobs/" + subFile.etag)
            fileDownloadTask.blobPathIncomplete =
                File(storageFolder, "blobs/" + subFile.etag + ".incomplete")
            fileDownloadTask.pointerPath = File(parentPointerPath, subFile.path)
            fileDownloadTask.downloadedSize =
                if (fileDownloadTask.blobPath!!.exists()) fileDownloadTask.blobPath!!.length() else (if (fileDownloadTask.blobPathIncomplete!!.exists()) fileDownloadTask.blobPathIncomplete!!.length() else 0)
            totalAndDownloadSize[0] += subFile.size
            totalAndDownloadSize[1] += fileDownloadTask.downloadedSize
            fileDownloadTasks.add(fileDownloadTask)
        }
        return fileDownloadTasks
    }


    companion object  {
        fun getCachePathRoot(modelDownloadPathRoot:String): String {
            return "$modelDownloadPathRoot/modelers"
        }

        fun getModelPath(modelsDownloadPathRoot: String, modelId: String): File {
            return File(modelsDownloadPathRoot, getLastFileName(modelId))
        }
    }

}
