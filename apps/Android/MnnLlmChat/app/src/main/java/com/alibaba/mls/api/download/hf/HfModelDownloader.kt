// Created by ruoyi.sjd on 2025/5/8.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mls.api.download.hf

import android.util.Log
import com.alibaba.mls.api.ApplicationProvider
import com.alibaba.mls.api.FileDownloadException
import com.alibaba.mls.api.hf.HfApiClient
import com.alibaba.mls.api.hf.HfApiClient.Companion.bestClient
import com.alibaba.mls.api.hf.HfFileMetadata
import com.alibaba.mls.api.hf.HfRepoInfo
import com.alibaba.mls.api.download.DownloadExecutor.Companion.executor
import com.alibaba.mls.api.download.DownloadFileUtils.createSymlink
import com.alibaba.mls.api.download.DownloadFileUtils.deleteDirectoryRecursively
import com.alibaba.mls.api.download.DownloadFileUtils.getLastFileName
import com.alibaba.mls.api.download.DownloadFileUtils.getPointerPathParent
import com.alibaba.mls.api.download.DownloadFileUtils.repoFolderName
import com.alibaba.mls.api.download.DownloadPausedException
import com.alibaba.mls.api.download.DownloadPersistentData.getMetaData
import com.alibaba.mls.api.download.DownloadPersistentData.saveMetaData
import com.alibaba.mls.api.download.FileDownloadTask
import com.alibaba.mls.api.download.hf.HfFileMetadataUtils.getFileMetadata
import com.alibaba.mls.api.download.ModelDownloadManager.Companion.TAG
import com.alibaba.mls.api.download.ModelFileDownloader
import com.alibaba.mls.api.download.ModelFileDownloader.FileDownloadListener
import com.alibaba.mls.api.download.ModelRepoDownloader
import com.alibaba.mls.api.source.ModelSources
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import java.io.File
import java.util.concurrent.TimeUnit

class HfModelDownloader(override var callback: ModelRepoDownloadCallback?,
                        override var cacheRootPath: String
) : ModelRepoDownloader() {

    private fun getHfApiClient(): HfApiClient {
        if (hfApiClient == null) {
            hfApiClient = bestClient
        }
        if (hfApiClient == null) {
            hfApiClient = HfApiClient(HfApiClient.HOST_DEFAULT)
        }
        return hfApiClient!!
    }
    private var hfApiClient: HfApiClient? = null
    private var metaInfoClient: OkHttpClient? = null

    override fun setListener(callback: ModelRepoDownloadCallback?) {
        this.callback = callback
    }

    override fun download(modelId: String) {
        getHfApiClient().getRepoInfo(modelId, "main", object :
            HfApiClient.RepoInfoCallback {
            override fun onSuccess(hfRepoInfo: HfRepoInfo?) {
                downloadHfRepo(hfRepoInfo!!)
            }

            override fun onFailure(error: String?) {
                callback?.onDownloadFailed(modelId, FileDownloadException("getRepoInfoFailed$error"))
            }
        })
    }

    fun downloadHfRepo(hfRepoInfo: HfRepoInfo) {
        Log.d(TAG, "DownloadStart " + hfRepoInfo.modelId + " host: " + getHfApiClient().host)
        executor!!.submit {
            callback?.onDownloadTaskAdded()
            downloadHfRepoInner(hfRepoInfo)
            callback?.onDownloadTaskRemoved()
        }
    }

    override suspend fun getRepoSize(modelId: String):Long {
        var totalSize = 0L
        withContext(Dispatchers.IO) {
            val result = getHfApiClient().apiService.getRepoInfo(modelId, "main")
            val response = kotlin.runCatching {result?.execute()}.getOrNull()
            if (response?.isSuccessful != true || response.body() == null) {
                return@withContext
            }
            val hfRepoInfo = response.body()!!
            val metaList = kotlin.runCatching {requestMetaDataList(hfRepoInfo)}.getOrNull()
            metaList?.forEach{
                totalSize += it?.size ?: 0
            }
        }
        return totalSize
    }

    private fun downloadHfRepoInner(hfRepoInfo: HfRepoInfo) {
        val folderLinkFile = File(cacheRootPath, getLastFileName(hfRepoInfo.modelId!!))
        if (folderLinkFile.exists()) {
            callback?.onDownloadFileFinished(hfRepoInfo.modelId!!, folderLinkFile.absolutePath)
            return
        }
        val modelDownloader = ModelFileDownloader()
        Log.d(TAG, "Repo SHA: " + hfRepoInfo.sha)

        val hasError = false
        val errorInfo = StringBuilder()

        val repoFolderName = repoFolderName(hfRepoInfo.modelId, "model")
        val storageFolder = File(cacheRootPath, repoFolderName)
        val parentPointerPath = getPointerPathParent(
            storageFolder,
            hfRepoInfo.sha!!
        )
        val downloadTaskList: List<FileDownloadTask>
        val totalAndDownloadSize = LongArray(2)
        try {
            downloadTaskList =
                collectTaskList(storageFolder, parentPointerPath, hfRepoInfo, totalAndDownloadSize)
        } catch (e: FileDownloadException) {
            callback?.onDownloadFailed(hfRepoInfo.modelId!!, e)
            return
        }
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
                        hfRepoInfo.modelId!!, "file", fileName,
                        totalAndDownloadSize[1], totalAndDownloadSize[0]
                    )
                    return pausedSet.contains(hfRepoInfo.modelId)
                }
            }
        try {
            for (fileDownloadTask in downloadTaskList) {
                modelDownloader.downloadFile(fileDownloadTask, fileDownloadListener)
            }
        } catch (e: DownloadPausedException) {
            pausedSet.remove(hfRepoInfo.modelId)
            callback?.onDownloadPaused(hfRepoInfo.modelId!!)
            return
        } catch (e: Exception) {
            callback?.onDownloadFailed(hfRepoInfo.modelId!!, e)
            return
        }
        if (!hasError) {
            val folderLinkPath = folderLinkFile.absolutePath
            createSymlink(parentPointerPath.toString(), folderLinkPath)
            callback?.onDownloadFileFinished(hfRepoInfo.modelId!!, folderLinkPath)
        } else {
            Log.e(
                TAG,
                "Errors occurred during download: $errorInfo"
            )
        }
    }

    @Throws(FileDownloadException::class)
    private fun collectTaskList(
        storageFolder: File,
        parentPointerPath: File,
        hfRepoInfo: HfRepoInfo,
        totalAndDownloadSize: LongArray
    ): List<FileDownloadTask> {
        var metaData: HfFileMetadata
        var metaDataList: List<HfFileMetadata?>? =
            getMetaData(ApplicationProvider.get(), hfRepoInfo.modelId!!)
        Log.d(
            TAG, "collectTaskList savedMetaDataList: " + (metaDataList?.size
                ?: "null")
        )
        val fileDownloadTasks: MutableList<FileDownloadTask> = ArrayList()
        metaDataList = requestMetaDataList(hfRepoInfo)
        saveMetaData(ApplicationProvider.get(), hfRepoInfo.modelId!!, metaDataList)
        for (i in hfRepoInfo.getSiblings().indices) {
            val subFile = hfRepoInfo.getSiblings()[i]
            metaData = metaDataList[i]!!
            val fileDownloadTask = FileDownloadTask()
            fileDownloadTask.relativePath = subFile.rfilename
            fileDownloadTask.fileMetadata = metaData
            fileDownloadTask.etag = metaData.etag
            fileDownloadTask.blobPath = File(storageFolder, "blobs/" + metaData.etag)
            fileDownloadTask.blobPathIncomplete =
                File(storageFolder, "blobs/" + metaData.etag + ".incomplete")
            fileDownloadTask.pointerPath = File(parentPointerPath, subFile.rfilename)
            fileDownloadTask.downloadedSize =
                if (fileDownloadTask.blobPath!!.exists()) fileDownloadTask.blobPath!!.length() else (if (fileDownloadTask.blobPathIncomplete!!.exists()) fileDownloadTask.blobPathIncomplete!!.length() else 0)
            totalAndDownloadSize[0] += metaData.size
            totalAndDownloadSize[1] += fileDownloadTask.downloadedSize
            fileDownloadTasks.add(fileDownloadTask)
        }
        return fileDownloadTasks
    }

    override fun getDownloadPath(modelId: String): File {
        return getModelPath(cacheRootPath, modelId)
    }

    @Throws(FileDownloadException::class)
    private fun requestMetaDataList(hfRepoInfo: HfRepoInfo): List<HfFileMetadata?> {
        val list: MutableList<HfFileMetadata?> = ArrayList()
        for (subFile in hfRepoInfo.getSiblings()) {
            val url =
                "https://" + getHfApiClient().host + "/" + hfRepoInfo.modelId + "/resolve/main/" + subFile.rfilename
            val metaData = getFileMetadata(metaInfoHttpClient, url)
            list.add(metaData)
        }
        return list
    }

    override fun deleteRepo(modelId: String) {
        val repoFolderName = repoFolderName(modelId, "model")
        val hfStorageFolder = File(cacheRootPath, repoFolderName)
        Log.d(TAG, "removeStorageFolder: " + hfStorageFolder.absolutePath)
        if (hfStorageFolder.exists()) {
            val result = deleteDirectoryRecursively(hfStorageFolder)
            if (!result) {
                Log.e(TAG, "remove storageFolder" + hfStorageFolder.absolutePath + " faield")
            }
        }
        val hfLinkFolder = this.getDownloadPath(modelId)
        Log.d(TAG, "removeHfLinkFolder: " + hfLinkFolder.absolutePath)
        hfLinkFolder.delete()
    }

    private val metaInfoHttpClient: OkHttpClient
        get() {
            if (metaInfoClient == null) {
                metaInfoClient = OkHttpClient.Builder()
                    .connectTimeout(30, TimeUnit.SECONDS)
                    .followRedirects(false)
                    .followSslRedirects(false)
                    .build()
            }
            return metaInfoClient!!
        }

    companion object  {
        fun getCachePathRoot(modelDownloadPathRoot:String): String {
            return modelDownloadPathRoot
        }

        fun getModelPath(cacheRootPath: String, modelId: String): File {
            val modelScopeId = ModelSources.get().config.getRepoConfig(modelId)!!.modelScopePath
            return File(cacheRootPath, getLastFileName(modelScopeId))
        }
    }
}
