// Created by ruoyi.sjd on 2025/5/8.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mls.api.download.ms
import android.util.Log
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
import com.alibaba.mls.api.source.ModelSources
import com.alibaba.mls.api.source.RepoConfig
import kotlinx.coroutines.Dispatchers
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

    override fun download(modelId: String) {
        downloadMsRepo(modelId)
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
        var result = 0L
        val repoConfig = ModelSources.get().config.getRepoConfig(modelId)
        val modelScopeId = repoConfig!!.repositoryPath()
        val split = modelScopeId.split("/".toRegex()).dropLastWhile { it.isEmpty() }.toTypedArray()
        if (split.size != 2) {
            return 0L
        }
        withContext(Dispatchers.IO) {
            runCatching {
                val msRepoInfo = msApiClient.apiService.getModelFiles(split[0], split[1]).execute().body()
                msRepoInfo?.Data?.Files?.forEach {
                    result += it.Size
                }
            }.getOrElse {
                Log.e(TAG, "getRepoSize: ", it)
            }
        }
        return result
    }

    private fun downloadMsRepo(modelId: String) {
        val repoConfig = ModelSources.get().config.getRepoConfig(modelId)
        val modelScopeId = repoConfig!!.repositoryPath()
        val split = modelScopeId.split("/".toRegex()).dropLastWhile { it.isEmpty() }.toTypedArray()
        if (split.size != 2) {
            callback?.onDownloadFailed(modelId, FileDownloadException("getRepoInfoFailed modelId format error: $modelId"))
        }
        msApiClient.apiService.getModelFiles(split[0], split[1])
            .enqueue(object : Callback<MsRepoInfo?> {
                override fun onResponse(call: Call<MsRepoInfo?>, response: Response<MsRepoInfo?>) {
                    executor!!.submit {
                        Log.d(
                            TAG,
                            "downloadMsRepoInner executor"
                        )
                        callback?.onDownloadTaskAdded()
                        downloadMsRepoInner(repoConfig, response.body()!!)
                        callback?.onDownloadTaskRemoved()
                    }
                }

                override fun onFailure(call: Call<MsRepoInfo?>, t: Throwable) {
                    callback?.onDownloadFailed(modelId, FileDownloadException("getRepoInfoFailed" + t.message))
                }
            })
    }

    private fun downloadMsRepoInner(repoConfig: RepoConfig, msRepoInfo: MsRepoInfo) {
        Log.d(TAG, "downloadMsRepoInner")
        val modelId = repoConfig.modelId
        val folderLinkFile =
            File(cacheRootPath, getLastFileName(repoConfig.repositoryPath()))
        if (folderLinkFile.exists()) {
            Log.d(TAG, "downloadMsRepoInner already exists")
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
        Log.d(TAG, "downloadMsRepoInner collectMsTaskList")
        downloadTaskList = collectMsTaskList(
            repoConfig,
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
        repoConfig: RepoConfig,
        storageFolder: File,
        parentPointerPath: File,
        msRepoInfo: MsRepoInfo,
        totalAndDownloadSize: LongArray
    ): List<FileDownloadTask> {
        val fileDownloadTasks: MutableList<FileDownloadTask> = ArrayList()
        for (i in msRepoInfo.Data.Files.indices) {
            val subFile = msRepoInfo.Data.Files[i]
            val fileDownloadTask = FileDownloadTask()
            if (subFile.Type == "tree") {
                continue
            }
            fileDownloadTask.relativePath = subFile.Path
            fileDownloadTask.fileMetadata = HfFileMetadata()
            fileDownloadTask.fileMetadata!!.location = String.format(
                "https://modelscope.cn/api/v1/models/%s/repo?FilePath=%s",
                repoConfig.repositoryPath(),
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
            val modelScopeId = ModelSources.get().config.getRepoConfig(modelId)!!.modelScopePath
            return File(modelsDownloadPathRoot, getLastFileName(modelScopeId))
        }
    }

}
