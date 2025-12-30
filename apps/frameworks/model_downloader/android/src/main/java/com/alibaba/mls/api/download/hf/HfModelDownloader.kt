// Created by ruoyi.sjd on 2025/5/8.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mls.api.download.hf

import android.util.Log
import com.alibaba.mls.api.ApplicationProvider
import com.alibaba.mls.api.FileDownloadException
import com.alibaba.mls.api.download.DownloadExecutor.Companion.executor
import com.alibaba.mls.api.download.DownloadFileUtils.createSymlink
import com.alibaba.mls.api.download.DownloadFileUtils.deleteDirectoryRecursively
import com.alibaba.mls.api.download.DownloadFileUtils.getLastFileName
import com.alibaba.mls.api.download.DownloadFileUtils.getPointerPathParent
import com.alibaba.mls.api.download.DownloadFileUtils.repoFolderName
import com.alibaba.mls.api.download.DownloadPausedException
import com.alibaba.mls.api.download.FileDownloadTask

import com.alibaba.mls.api.download.ModelFileDownloader
import com.alibaba.mls.api.download.ModelFileDownloader.FileDownloadListener
import com.alibaba.mls.api.download.ModelRepoDownloader
import com.alibaba.mls.api.download.hf.HfFileMetadataUtils.getFileMetadata
import com.alibaba.mls.api.HfApiClient
import com.alibaba.mls.api.HfApiClient.Companion.bestClient
import com.alibaba.mls.api.HfFileMetadata
import com.alibaba.mls.api.HfRepoInfo
import com.alibaba.mls.api.download.TimeUtils
import com.alibaba.mls.api.download.DownloadCoroutineManager
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import java.io.File
import java.util.concurrent.TimeUnit

class HfModelDownloader(override var callback: ModelRepoDownloadCallback?,
                        override var cacheRootPath: String
) : ModelRepoDownloader() {

    companion object {
        private const val TAG = "HfModelDownloader"

        fun getCachePathRoot(modelDownloadPathRoot: String): String {
            return "$modelDownloadPathRoot/hf"
        }

        fun getModelPath(modelsDownloadPathRoot: String, modelId: String): File {
            return File(modelsDownloadPathRoot, getLastFileName(modelId))
        }
    }

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

    private fun hfModelId(modelId: String): String {
        return modelId.replace("HuggingFace/", "")
    }

    /**
     * Unified method to fetch repo information from HuggingFace API
     * @param modelId the model ID to fetch info for
     * @param calculateSize whether to calculate repo size (requires additional network requests)
     * @return HfRepoInfo object
     * @throws FileDownloadException if failed to fetch repo info
     */
    private suspend fun fetchRepoInfo(modelId: String, calculateSize: Boolean = false): HfRepoInfo {
        return withContext(DownloadCoroutineManager.downloadDispatcher) {
            runCatching {
                val hfModelId = hfModelId(modelId)
                val response = getHfApiClient().getRepoTree(hfModelId, "main").execute()
                if (response?.isSuccessful == true && response.body() != null) {
                    val repoInfo = response.body()!!
                    // Call onRepoInfo callback with repo metadata
                    val lastModified = System.currentTimeMillis() // Simplified
                    val repoSize = if (calculateSize) {
                        calculateRepoSize(repoInfo)
                    } else {
                        0L // Will be calculated separately when needed
                    }
                    callback?.onRepoInfo(modelId, lastModified, repoSize)
                    repoInfo
                } else {
                    val errorMsg = if (response?.isSuccessful == false) {
                        "API request failed with code ${response.code()}: ${response.message()}"
                    } else {
                        "API response was null or empty"
                    }
                    throw FileDownloadException("Failed to fetch repo info for $modelId: $errorMsg")
                }
            }.getOrElse { exception ->
                Log.e(TAG, "Failed to fetch repo info for $modelId", exception)
                throw FileDownloadException("Failed to fetch repo info for $modelId: ${exception.message}")
            }
        }
    }

    /**
     * Calculate total size of all files in the repo
     */
    private suspend fun calculateRepoSize(hfRepoInfo: HfRepoInfo): Long {
        return withContext(DownloadCoroutineManager.downloadDispatcher) {
            runCatching {
                val metaList = requestMetaDataList(hfRepoInfo)
                metaList.sumOf { it?.size ?: 0L }
            }.getOrElse { 0L }
        }
    }

    override fun download(modelId: String) {
        DownloadCoroutineManager.launchDownload {
            try {
                val repoInfo = fetchRepoInfo(modelId)
                downloadHfRepo(repoInfo)
            } catch (e: FileDownloadException) {
                callback?.onDownloadFailed(modelId, e)
            }
        }
    }

    override suspend fun checkUpdate(modelId: String) {
        try {
            fetchRepoInfo(modelId)
        } catch (e: FileDownloadException) {
            Log.e(TAG, "Failed to check update for $modelId", e)
        }
    }

    fun downloadHfRepo(hfRepoInfo: HfRepoInfo) {
        Log.d(TAG, "DownloadStart " + hfRepoInfo.modelId + " host: " + getHfApiClient().host)
        executor!!.submit {
            callback?.onDownloadTaskAdded()
            downloadHfRepoInner(hfRepoInfo)
            callback?.onDownloadTaskRemoved()
        }
    }

    override suspend fun getRepoSize(modelId: String): Long {
        return withContext(DownloadCoroutineManager.downloadDispatcher) {
            runCatching {
                val repoInfo = fetchRepoInfo(modelId, calculateSize = true)
                // Size was already calculated in fetchRepoInfo, but we can also calculate it directly
                calculateRepoSize(repoInfo)
            }.getOrElse { exception ->
                Log.e(TAG, "Failed to get repo size for $modelId", exception)
                // Try to get file_size from saved market data as fallback
                try {
                    val marketSize = com.alibaba.mls.api.download.DownloadPersistentData.getMarketSizeTotal(ApplicationProvider.get(), modelId)
                    if (marketSize > 0) {
                        Log.d(TAG, "Using saved market size as fallback for $modelId: $marketSize")
                        return@withContext marketSize
                    }
                } catch (e: Exception) {
                    Log.w(TAG, "Failed to get saved market size for $modelId", e)
                }
                0L
            }
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
                    return pausedSet.contains(hfRepoInfo.modelId!!)
                }
            }
        try {
            for (fileDownloadTask in downloadTaskList) {
                modelDownloader.downloadFile(fileDownloadTask, fileDownloadListener)
            }
        } catch (e: DownloadPausedException) {
            pausedSet.remove(hfRepoInfo.modelId!!)
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
        val fileDownloadTasks: MutableList<FileDownloadTask> = ArrayList()
        var metaDataList = requestMetaDataList(hfRepoInfo)
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
        val hfModelId = hfModelId(modelId)
        val repoFolderName = repoFolderName(hfModelId, "model")
        val hfStorageFolder = File(cacheRootPath, repoFolderName)
        Log.d(TAG, "removeStorageFolder: " + hfStorageFolder.absolutePath)
        if (hfStorageFolder.exists()) {
            val result = deleteDirectoryRecursively(hfStorageFolder)
            if (!result) {
                Log.e(TAG, "remove storageFolder" + hfStorageFolder.absolutePath + " failed")
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
}
