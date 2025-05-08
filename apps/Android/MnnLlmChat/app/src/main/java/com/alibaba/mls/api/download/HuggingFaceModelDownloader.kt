// Created by ruoyi.sjd on 2025/5/8.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mls.api.download

import android.util.Log
import com.alibaba.mls.api.ApplicationProvider
import com.alibaba.mls.api.FileDownloadException
import com.alibaba.mls.api.HfApiClient
import com.alibaba.mls.api.HfFileMetadata
import com.alibaba.mls.api.HfRepoInfo
import com.alibaba.mls.api.download.DownloadExecutor.Companion.executor
import com.alibaba.mls.api.download.DownloadFileUtils.createSymlink
import com.alibaba.mls.api.download.DownloadFileUtils.deleteDirectoryRecursively
import com.alibaba.mls.api.download.DownloadFileUtils.getLastFileName
import com.alibaba.mls.api.download.DownloadFileUtils.getPointerPathParent
import com.alibaba.mls.api.download.DownloadFileUtils.repoFolderName
import com.alibaba.mls.api.download.DownloadPersistentData.getMetaData
import com.alibaba.mls.api.download.DownloadPersistentData.saveMetaData
import com.alibaba.mls.api.download.HfFileMetadataUtils.getFileMetadata
import com.alibaba.mls.api.download.ModelDownloadManager.Companion.TAG
import com.alibaba.mls.api.download.ModelFileDownloader.FileDownloadListener
import okhttp3.OkHttpClient
import java.io.File
import java.util.concurrent.TimeUnit

class HuggingFaceModelDownloader(private val manager: ModelDownloadManager,
                                 override var callback: ModelRepoDownloadCallback?,
                                 override var cacheRootPath: String
) : ModelRepoDownloader() {

    private val hfApiClient by lazy { manager.getHfApiClient() }
    private var metaInfoClient: OkHttpClient? = null

    override fun setListener(callback: ModelRepoDownloadCallback?) {
        this.callback = callback
    }

    override fun download(modelId: String) {
        manager.getHfApiClient().getRepoInfo(modelId, "main", object :
            HfApiClient.RepoInfoCallback {
            override fun onSuccess(hfRepoInfo: HfRepoInfo?) {
                downloadHfRepo(hfRepoInfo!!)
            }

            override fun onFailure(error: String?) {
                manager.setDownloadFailed(modelId, FileDownloadException("getRepoInfoFailed$error"))
            }
        })
    }



    fun downloadHfRepo(hfRepoInfo: HfRepoInfo) {
        Log.d(TAG, "DownloadStart " + hfRepoInfo.modelId + " host: " + hfApiClient.host)
        executor!!.submit {
            callback?.onDownloadTaskAdded()
            downloadHfRepoInner(hfRepoInfo)
            callback?.onDownloadTaskRemoved()
        }
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
        metaDataList = requestMedataDataList(hfRepoInfo)
        saveMetaData(ApplicationProvider.get(), hfRepoInfo.modelId!!, metaDataList)
        for (i in hfRepoInfo.getSiblings().indices) {
            val subFile = hfRepoInfo.getSiblings()[i]
            metaData = metaDataList[i]!!
            val fileDownloadTask = FileDownloadTask()
            fileDownloadTask.relativePath = subFile.rfilename
            fileDownloadTask.hfFileMetadata = metaData
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

    override fun resume(modelId: String) {
        manager.startDownload(modelId)
    }

    override fun cancel(modelId: String) {
        manager.deleteRepo(modelId)
    }

    override fun getDownloadPath(modelId: String): File {
        return manager.getHfDownloadModelPath(modelId)
    }

    @Throws(FileDownloadException::class)
    private fun requestMedataDataList(hfRepoInfo: HfRepoInfo): List<HfFileMetadata?> {
        val list: MutableList<HfFileMetadata?> = ArrayList()
        for (subFile in hfRepoInfo.getSiblings()) {
            val url =
                "https://" + hfApiClient.host + "/" + hfRepoInfo.modelId + "/resolve/main/" + subFile.rfilename
            val metaData = getFileMetadata(metaInfoHttpClient, url)
            list.add(metaData)
        }
        return list
    }

    override fun isDownloaded(modelId: String): Boolean {
        return manager.getHfDownloadModelPath(modelId).exists()
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
        val hfLinkFolder = this.getDownloadedFile(modelId)
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
}
