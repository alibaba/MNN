// Created by ruoyi.sjd on 2025/5/8.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mls.api.download.ml
import android.util.Log
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
import com.alibaba.mls.api.ml.MlApiClient
import com.alibaba.mls.api.ml.MlRepoInfo
import com.alibaba.mls.api.source.ModelSources
import com.alibaba.mls.api.source.RepoConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class MLModelDownloader(override var callback: ModelRepoDownloadCallback?,
                        cacheRootPath: String
) : ModelRepoDownloader() {
    override var cacheRootPath: String = getCachePathRoot(cacheRootPath)
    private var mlApiClient: MlApiClient = MlApiClient()
    override fun setListener(callback: ModelRepoDownloadCallback?) {
        this.callback = callback
    }

    override fun download(modelId: String) {
        DownloadExecutor.executeScope.launch {
            downloadRepo(modelId)
        }
    }

    override fun getDownloadPath(modelId: String): File {
        return getModelPath(cacheRootPath, modelId)
    }

    override fun deleteRepo(modelId: String) {
        val msModelId = ModelSources.get().config.getRepoConfig(modelId)!!.modelScopePath
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

    override suspend fun getRepoSize(modelId: String):Long {
        var result: Long
        val repoConfig = ModelSources.get().config.getRepoConfig(modelId)
        val modelScopeId = repoConfig!!.repositoryPath()
        val split = modelScopeId.split("/".toRegex()).dropLastWhile { it.isEmpty() }.toTypedArray()
        if (split.size != 2) {
            return 0L
        }
        result = withContext(Dispatchers.IO) {
            runCatching {
                val mlRepoInfo = mlApiClient.apiService.getModelFiles(split[0], split[1]).execute().body()
                mlRepoInfo?.data?.tree?.sumOf { it.size } ?: 0L
            }.getOrElse {
                Log.e(TAG, "getRepoSize: ", it)
                return@withContext 0L
            }
        }
        return result
    }

    private fun downloadRepo(modelId: String): MlRepoInfo? {
        val repoConfig = ModelSources.get().config.getRepoConfig(modelId)
        val modelScopeId = repoConfig!!.repositoryPath()
        val split = modelScopeId.split("/".toRegex()).dropLastWhile { it.isEmpty() }.toTypedArray()
        if (split.size != 2) {
            callback?.onDownloadFailed(modelId, FileDownloadException("getRepoInfoFailed modelId format error: $modelId"))
        }
        val modelInfo:MlRepoInfo? = kotlin.runCatching {
            mlApiClient.apiService.getModelFiles(split[0], split[1]).execute().body() }
            .getOrElse {
                callback?.onDownloadFailed(modelId, FileDownloadException("getRepoInfoFailed" + it.message))
                null}
        if (modelInfo != null) {
            callback?.onDownloadTaskAdded()
            downloadMlRepoInner(repoConfig, modelInfo)
            callback?.onDownloadTaskRemoved()
        }
        return modelInfo
    }

    private fun downloadMlRepoInner(repoConfig: RepoConfig, mlRepoInfo: MlRepoInfo) {
        Log.d(TAG, "downloadMlRepoInner")
        val modelId = repoConfig.modelId
        val folderLinkFile =
            File(cacheRootPath, getLastFileName(repoConfig.repositoryPath()))
        if (folderLinkFile.exists()) {
            Log.d(TAG, "downloadMlRepoInner already exists")
            callback?.onDownloadFileFinished(modelId, folderLinkFile.absolutePath)
            return
        }
        val modelDownloader = ModelFileDownloader()
        val hasError = false
        val errorInfo = StringBuilder()
        val repoFolderName = repoFolderName(repoConfig.repositoryPath(), "model")
        val storageFolder = File(this.cacheRootPath, repoFolderName)
        val parentPointerPath = getPointerPathParent(storageFolder, "_no_sha_")
        val downloadTaskList: List<FileDownloadTask>
        val totalAndDownloadSize = LongArray(2)
        Log.d(TAG, "downloadMlRepoInner collectMsTaskList")
        downloadTaskList = collectTaskList(
            repoConfig,
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
        repoConfig: RepoConfig,
        storageFolder: File,
        parentPointerPath: File,
        mlRepoInfo: MlRepoInfo,
        totalAndDownloadSize: LongArray
    ): List<FileDownloadTask> {
        val fileDownloadTasks: MutableList<FileDownloadTask> = ArrayList()
        for (i in mlRepoInfo.data.tree.indices) {
            val subFile = mlRepoInfo.data.tree[i]
            val fileDownloadTask = FileDownloadTask()
            fileDownloadTask.relativePath = subFile.path
            fileDownloadTask.fileMetadata = HfFileMetadata()
            fileDownloadTask.fileMetadata!!.location = String.format(
                "https://modelers.cn/coderepo/web/v1/file/%s/main/media/%s",
                repoConfig.repositoryPath(),
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
            val modelScopeId = ModelSources.get().config.getRepoConfig(modelId)!!.modelScopePath
            return File(modelsDownloadPathRoot, getLastFileName(modelScopeId))
        }
    }

}
