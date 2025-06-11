// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mls.api.download

import android.Manifest
import android.annotation.SuppressLint
import android.app.Activity
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.util.Log
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.alibaba.mls.api.ApplicationProvider
import com.alibaba.mls.api.HfApiClient
import com.alibaba.mls.api.HfApiClient.Companion.bestClient
import com.alibaba.mls.api.HfApiClient.RepoInfoCallback
import com.alibaba.mls.api.HfApiException
import com.alibaba.mls.api.HfFileMetadata
import com.alibaba.mls.api.HfRepoInfo
import com.alibaba.mls.api.download.DownloadExecutor.Companion.executor
import com.alibaba.mls.api.download.DownloadFileUtils.createSymlink
import com.alibaba.mls.api.download.DownloadFileUtils.deleteDirectoryRecursively
import com.alibaba.mls.api.download.DownloadFileUtils.getLastFileName
import com.alibaba.mls.api.download.DownloadFileUtils.getPointerPathParent
import com.alibaba.mls.api.download.DownloadFileUtils.repoFolderName
import com.alibaba.mls.api.download.DownloadPersistentData.getDownloadSizeSaved
import com.alibaba.mls.api.download.DownloadPersistentData.getDownloadSizeTotal
import com.alibaba.mls.api.download.DownloadPersistentData.getMetaData
import com.alibaba.mls.api.download.DownloadPersistentData.removeProgress
import com.alibaba.mls.api.download.DownloadPersistentData.saveDownloadSizeSaved
import com.alibaba.mls.api.download.DownloadPersistentData.saveDownloadSizeTotal
import com.alibaba.mls.api.download.DownloadPersistentData.saveMetaData
import com.alibaba.mls.api.download.HfFileMetadataUtils.getFileMetadata
import com.alibaba.mls.api.download.ModelFileDownloader.FileDownloadListener
import com.alibaba.mls.api.ms.MsApiClient
import com.alibaba.mls.api.ms.MsRepoInfo
import com.alibaba.mls.api.source.ModelSources
import com.alibaba.mls.api.source.RepoConfig
import com.alibaba.mnnllm.android.utils.FileUtils
import com.alibaba.mnnllm.android.utils.FileUtils.clearMmapCache
import okhttp3.OkHttpClient
import retrofit2.Call
import retrofit2.Callback
import retrofit2.Response
import java.io.File
import java.util.Collections
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import kotlin.collections.set

class ModelDownloadManager (private val context: Context) {
    private var downloadListener: DownloadListener? = null
    private val cachePath = context.filesDir.absolutePath + "/.mnnmodels"
    private val modelScopeCachePath = cachePath + "/modelscope"

    private var hfApiClient: HfApiClient? = null
    private var msApiClient: MsApiClient? = null
        get() {
            if (field == null) {
                field = MsApiClient()
            }
            return field
        }

    private var metaInfoClient: OkHttpClient? = null
    private val downloadInfoMap = HashMap<String, DownloadInfo>()
    private val pausedSet: MutableSet<String?> = Collections.synchronizedSet(HashSet())
    private val foregroundSerivceIntent =
        Intent(context.applicationContext, DownlodForegroundService::class.java)

    private val activeDownloadCount =
        AtomicInteger(0)

    private var foregroundServiceStarted = false

    fun setListener(downloadListener: DownloadListener?) {
        this.downloadListener = downloadListener
    }

    fun getDownloadPath(modelId: String): File {
        return if (ModelSources.get().remoteSourceType == ModelSources.ModelSourceType.HUGGING_FACE) {
            getHfDownloadModelPath(modelId)
        } else {
            getMsModelPath(modelId)
        }
    }

    private fun getDownloadedFile(modelId: String): File? {
        val file = getDownloadPath(modelId)
        if (file.exists()) {
            return file
        } else if (getHfDownloadModelPath(modelId).exists()) {
            return getHfDownloadModelPath(modelId)
        } else if (getMsModelPath(modelId).exists()) {
            return getMsModelPath(modelId)
        }
        return null
    }

    private fun getHfDownloadModelPath(modelId: String): File {
        return File(cachePath, getLastFileName(modelId)!!)
    }

    private fun getMsModelPath(modelId: String): File {
        val modelScopeId = ModelSources.get().config.getRepoConfig(modelId)!!.modelScopePath
        return File(this.modelScopeCachePath, getLastFileName(modelScopeId)!!)
    }

    fun pauseDownload(modelId: String) {
        if (!getDownloadInfo(modelId).isDownloadingOrPreparing()) {
            return
        }
        pausedSet.add(modelId)
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

    fun startDownload(modelId: String) {
        if (downloadListener != null) {
            downloadListener!!.onDownloadStart(modelId)
        }
        this.updateDownloadingProgress(modelId, "Preparing", null, 0, 10)
        if (ModelSources.get().remoteSourceType == ModelSources.ModelSourceType.HUGGING_FACE) {
            getHfApiClient().getRepoInfo(modelId, "main", object : RepoInfoCallback {
                override fun onSuccess(hfRepoInfo: HfRepoInfo?) {
                    downloadHfRepo(hfRepoInfo!!)
                }

                override fun onFailure(error: String?) {
                    setDownloadFailed(modelId, HfApiException("getRepoInfoFailed$error"))
                }
            })
        } else {
            downloadMsRepo(modelId)
        }
    }

    private fun downloadMsRepo(modelId: String) {
        val repoConfig = ModelSources.get().config.getRepoConfig(modelId)
        val modelScopeId = repoConfig!!.repositoryPath()
        val split = modelScopeId.split("/".toRegex()).dropLastWhile { it.isEmpty() }.toTypedArray()
        if (split.size != 2) {
            setDownloadFailed(
                modelId,
                HfApiException("getRepoInfoFailed modelId format error: $modelId")
            )
        }
        msApiClient!!.apiService.getModelFiles(split[0], split[1])
            .enqueue(object : Callback<MsRepoInfo?> {
                override fun onResponse(call: Call<MsRepoInfo?>, response: Response<MsRepoInfo?>) {
                    executor!!.submit {
                        Log.d(
                            TAG,
                            "downloadMsRepoInner executor"
                        )
                        onDownloadTaskAdded(activeDownloadCount.incrementAndGet())
                        downloadMsRepoInner(repoConfig, response.body()!!)
                        onDownloadTaskRemoved(activeDownloadCount.decrementAndGet())
                    }
                }

                override fun onFailure(call: Call<MsRepoInfo?>, t: Throwable) {
                    setDownloadFailed(modelId, HfApiException("getRepoInfoFailed" + t.message))
                }
            })
    }

    @SuppressLint("DefaultLocale")
    private fun calculateDownloadSpeed(downloadInfo: DownloadInfo, currentDownloadSize: Long) {
        val currentTime = System.currentTimeMillis()
        val lastLogTime = downloadInfo.lastLogTime
        val lastDownloadSize = downloadInfo.savedSize
        if (lastLogTime == 0L || lastDownloadSize == 0L) {
            downloadInfo.lastLogTime = currentTime
            downloadInfo.savedSize = currentDownloadSize
            downloadInfo.speedInfo = "0.00K/s"
        } else if (currentTime - lastLogTime >= 1000) {
            val timeDiff = currentTime - lastLogTime
            val speedBytesPerSecond = ((currentDownloadSize - lastDownloadSize) * 1000.0) / timeDiff
            downloadInfo.lastLogTime = currentTime
            downloadInfo.savedSize = currentDownloadSize
            if (speedBytesPerSecond >= 1024 * 1024) {
                downloadInfo.speedInfo =
                    String.format("%.2fM/s", speedBytesPerSecond / (1024 * 1024))
            } else if (speedBytesPerSecond >= 1024) {
                downloadInfo.speedInfo = String.format("%.2fK/s", speedBytesPerSecond / 1024)
            } else {
                downloadInfo.speedInfo = String.format("%.2fB/s", speedBytesPerSecond)
            }
        }
    }

    private fun downloadMsRepoInner(repoConfig: RepoConfig, msRepoInfo: MsRepoInfo) {
        Log.d(TAG, "downloadMsRepoInner")
        val modelId = repoConfig.modelId
        val folderLinkFile =
            File(this.modelScopeCachePath, getLastFileName(repoConfig.repositoryPath())!!)
        if (folderLinkFile.exists()) {
            Log.d(TAG, "downloadMsRepoInner already exists")
            setDownloadFinished(repoConfig.modelId, folderLinkFile.absolutePath)
            return
        }
        val modelDownloader = ModelFileDownloader()
        val hasError = false
        val errorInfo = StringBuilder()
        val repoFolderName = repoFolderName(repoConfig.repositoryPath(), "model")
        val storageFolder = File(this.modelScopeCachePath, repoFolderName)
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
            object :FileDownloadListener {
                override fun onDownloadDelta(
                    fileName: String?,
                    downloadedBytes: Long,
                    totalBytes: Long,
                    delta: Long
                ): Boolean {
                    totalAndDownloadSize[1] += delta
                    updateDownloadingProgress(
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
            setDownloadPaused(modelId)
            return
        } catch (e: Exception) {
            setDownloadFailed(modelId, e)
            return
        }
        if (!hasError) {
            val folderLinkPath = folderLinkFile.absolutePath
            createSymlink(parentPointerPath.toString(), folderLinkPath)
            setDownloadFinished(modelId, folderLinkPath)
        } else {
            Log.e(
                TAG,
                "Errors occurred during download: $errorInfo"
            )
        }
    }

    fun getDownloadInfo(modelId: String): DownloadInfo {
        if (!downloadInfoMap.containsKey(modelId)) {
            val downloadInfo = DownloadInfo()
            if (getDownloadedFile(modelId) != null) {
                downloadInfo.downlodaState = RepoDownloadSate.COMPLETED
                downloadInfo.totalSize = getDownloadSizeTotal(ApplicationProvider.get(), modelId)
                downloadInfo.progress = 1.0
            } else if (getDownloadSizeTotal(ApplicationProvider.get(), modelId) > 0) {
                val totalSize = getDownloadSizeTotal(ApplicationProvider.get(), modelId)
                val savedSize = getDownloadSizeSaved(ApplicationProvider.get(), modelId)
                downloadInfo.totalSize = totalSize
                downloadInfo.savedSize = savedSize
                downloadInfo.progress = savedSize.toDouble() / totalSize
                downloadInfo.downlodaState = RepoDownloadSate.PAUSED
            } else {
                downloadInfo.downlodaState = RepoDownloadSate.NOT_START
                downloadInfo.progress = 0.0
            }
            downloadInfoMap[modelId] = downloadInfo
        }
        return downloadInfoMap[modelId]!!
    }

    private fun setDownloadFinished(modelId: String, path: String) {
        val downloadInfo = getDownloadInfo(modelId)
        downloadInfo.downlodaState = RepoDownloadSate.COMPLETED
        if (downloadListener != null) {
            downloadListener!!.onDownloadFinished(modelId, path)
        }
    }

    private fun setDownloadPaused(modelId: String) {
        val downloadInfo = getDownloadInfo(modelId)
        downloadInfo.downlodaState = RepoDownloadSate.PAUSED
        if (downloadListener != null) {
            downloadListener!!.onDownloadPaused(modelId)
        }
    }

    private fun onDownloadTaskAdded(count: Int) {
        if (count == 1) {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                if (ContextCompat.checkSelfPermission(
                        context,
                        Manifest.permission.POST_NOTIFICATIONS
                    )
                    != PackageManager.PERMISSION_GRANTED
                ) {
                    ActivityCompat.requestPermissions(
                        context as Activity,
                        arrayOf(Manifest.permission.POST_NOTIFICATIONS),
                        REQUEST_CODE_POST_NOTIFICATIONS
                    )
                } else {
                    ApplicationProvider.get().startForegroundService(foregroundSerivceIntent)
                    foregroundServiceStarted = true
                }
            }
        }
    }

    private fun onDownloadTaskRemoved(count: Int) {
        if (count == 0) {
            ApplicationProvider.get().stopService(foregroundSerivceIntent)
            foregroundServiceStarted = false
        }
    }

    private fun setDownloadFailed(modelId: String, e: Exception) {
        Log.e(TAG, "onDownloadFailed: $modelId", e)
        val downloadInfo = getDownloadInfo(modelId)
        downloadInfo.downlodaState = RepoDownloadSate.FAILED
        downloadInfo.errorMessage = e.message
        if (downloadListener != null) {
            downloadListener!!.onDownloadFailed(modelId, e)
        }
    }

    private fun updateDownloadingProgress(
        modelId: String?,
        stage: String,
        currentFile: String?,
        saved: Long,
        total: Long
    ) {
        if (!downloadInfoMap.containsKey(modelId)) {
            val downloadInfo = DownloadInfo()
            downloadInfoMap[modelId!!] = downloadInfo
        }
        val downloadInfo = checkNotNull(downloadInfoMap[modelId])
        downloadInfo.progress = saved.toDouble() / total
        calculateDownloadSpeed(downloadInfo, saved)
        downloadInfo.totalSize = total
        downloadInfo.progressStage = stage
        downloadInfo.currentFile = currentFile
        downloadInfo.downlodaState = if (stage == "Preparing") {
            RepoDownloadSate.PREPARING
        }  else {
            RepoDownloadSate.DOWNLOADING
        }
        saveDownloadSizeTotal(ApplicationProvider.get(), modelId, total)
        saveDownloadSizeSaved(ApplicationProvider.get(), modelId, saved)
        if (downloadListener != null) {
            downloadListener!!.onDownloadProgress(modelId!!, downloadInfo)
        }
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

    @Throws(HfApiException::class)
    private fun requestMedataDataList(hfRepoInfo: HfRepoInfo): List<HfFileMetadata?> {
        val list: MutableList<HfFileMetadata?> = ArrayList()
        for (subFile in hfRepoInfo.getSiblings()) {
            val url =
                "https://" + hfApiClient!!.host + "/" + hfRepoInfo.modelId + "/resolve/main/" + subFile.rfilename
            val metaData = getFileMetadata(metaInfoHttpClient, url)
            list.add(metaData)
        }
        return list
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
            if (subFile.Type == "tree") {
                continue
            }
            val fileDownloadTask = FileDownloadTask()
            fileDownloadTask.relativePath = subFile.Path
            fileDownloadTask.hfFileMetadata = HfFileMetadata()
            if (ModelSources.get().remoteSourceType == ModelSources.ModelSourceType.MODELERS) {
                fileDownloadTask.hfFileMetadata!!.location = String.format(
                    "https://modelers.cn/coderepo/web/v1/file/%s/main/media/%s",
                    repoConfig.repositoryPath(),
                    subFile.Path
                )
            } else {
                fileDownloadTask.hfFileMetadata!!.location = String.format(
                    "https://modelscope.cn/api/v1/models/%s/repo?FilePath=%s",
                    repoConfig.repositoryPath(),
                    subFile.Path
                )
            }
            fileDownloadTask.hfFileMetadata!!.size = subFile.Size
            fileDownloadTask.hfFileMetadata!!.etag = subFile.Sha256
            fileDownloadTask.blobPath = File(storageFolder, "blobs/" + subFile.Path)
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

    @Throws(HfApiException::class)
    private fun collectTaskList(
        storageFolder: File,
        parentPointerPath: File,
        hfRepoInfo: HfRepoInfo,
        totalAndDownloadSize: LongArray
    ): List<FileDownloadTask> {
        var metaData: HfFileMetadata
        var metaDataList: List<HfFileMetadata?>? =
            getMetaData(ApplicationProvider.get(), hfRepoInfo.modelId)
        Log.d(
            TAG, "collectTaskList savedMetaDataList: " + (metaDataList?.size
                ?: "null")
        )
        val fileDownloadTasks: MutableList<FileDownloadTask> = ArrayList()
        metaDataList = requestMedataDataList(hfRepoInfo)
        saveMetaData(ApplicationProvider.get(), hfRepoInfo.modelId, metaDataList)
        for (i in hfRepoInfo.getSiblings().indices) {
            val subFile = hfRepoInfo.getSiblings()[i]
            metaData = metaDataList[i]!!
            val fileDownloadTask = FileDownloadTask()
            fileDownloadTask.relativePath = subFile.rfilename
            fileDownloadTask.hfFileMetadata = metaData
            fileDownloadTask.blobPath = File(storageFolder, "blobs/" + metaData.etag)
            fileDownloadTask.blobPathIncomplete =
                File(storageFolder, "blobs/" + metaData.etag + ".incomplete")
            fileDownloadTask.pointerPath = File(parentPointerPath, subFile.rfilename!!)
            fileDownloadTask.downloadedSize =
                if (fileDownloadTask.blobPath!!.exists()) fileDownloadTask.blobPath!!.length() else (if (fileDownloadTask.blobPathIncomplete!!.exists()) fileDownloadTask.blobPathIncomplete!!.length() else 0)
            totalAndDownloadSize[0] += metaData.size
            totalAndDownloadSize[1] += fileDownloadTask.downloadedSize
            fileDownloadTasks.add(fileDownloadTask)
        }
        return fileDownloadTasks
    }

    fun downloadHfRepo(hfRepoInfo: HfRepoInfo) {
        Log.d(TAG, "DownloadStart " + hfRepoInfo.modelId + " host: " + getHfApiClient().host)
        executor!!.submit {
            onDownloadTaskAdded(activeDownloadCount.incrementAndGet())
            downloadHfRepoInner(hfRepoInfo)
            onDownloadTaskRemoved(activeDownloadCount.decrementAndGet())
        }
    }

    private fun downloadHfRepoInner(hfRepoInfo: HfRepoInfo) {
        val folderLinkFile = File(cachePath, getLastFileName(hfRepoInfo.modelId)!!)
        if (folderLinkFile.exists()) {
            setDownloadFinished(hfRepoInfo.modelId!!, folderLinkFile.absolutePath)
            return
        }
        val modelDownloader = ModelFileDownloader()
        Log.d(TAG, "Repo SHA: " + hfRepoInfo.sha)

        val hasError = false
        val errorInfo = StringBuilder()

        val repoFolderName = repoFolderName(hfRepoInfo.modelId, "model")
        val storageFolder = File(cachePath, repoFolderName)
        val parentPointerPath = getPointerPathParent(
            storageFolder,
            hfRepoInfo.sha!!
        )
        val downloadTaskList: List<FileDownloadTask>
        val totalAndDownloadSize = LongArray(2)
        try {
            downloadTaskList =
                collectTaskList(storageFolder, parentPointerPath, hfRepoInfo, totalAndDownloadSize)
        } catch (e: HfApiException) {
            setDownloadFailed(hfRepoInfo.modelId!!, e)
            return
        }
        val fileDownloadListener =
            object :FileDownloadListener {
                override fun onDownloadDelta(
                    fileName: String?,
                    downloadedBytes: Long,
                    totalBytes: Long,
                    delta: Long
                ): Boolean {
                    totalAndDownloadSize[1] += delta
                    updateDownloadingProgress(
                        hfRepoInfo.modelId, "file", fileName,
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
            setDownloadPaused(hfRepoInfo.modelId!!)
            return
        } catch (e: Exception) {
            setDownloadFailed(hfRepoInfo.modelId!!, e)
            return
        }
        if (!hasError) {
            val folderLinkPath = folderLinkFile.absolutePath
            createSymlink(parentPointerPath.toString(), folderLinkPath)
            setDownloadFinished(hfRepoInfo.modelId!!, folderLinkPath)
        } else {
            Log.e(
                TAG,
                "Errors occurred during download: $errorInfo"
            )
        }
    }

    fun removeDownload(modelId: String) {
        val repoFolderName = repoFolderName(modelId, "model")
        val hfStorageFolder = File(cachePath, repoFolderName)
        Log.d(TAG, "removeStorageFolder: " + hfStorageFolder.absolutePath)
        if (hfStorageFolder.exists()) {
            val result = deleteDirectoryRecursively(hfStorageFolder)
            if (!result) {
                Log.e(TAG, "remove storageFolder" + hfStorageFolder.absolutePath + " faield")
            }
        }
        val msModelId = ModelSources.get().config.getRepoConfig(modelId)!!.modelScopePath
        val msRepoFolderName = repoFolderName(msModelId, "model")
        val msStorageFolder = File(this.modelScopeCachePath, msRepoFolderName)
        Log.d(TAG, "removeStorageFolder: " + msStorageFolder.absolutePath)
        if (msStorageFolder.exists()) {
            val result = deleteDirectoryRecursively(msStorageFolder)
            if (!result) {
                Log.e(TAG, "remove storageFolder" + msStorageFolder.absolutePath + " faield")
            }
        }
        removeProgress(ApplicationProvider.get(), modelId)
        val hfLinkFolder = this.getHfDownloadModelPath(modelId)
        Log.d(TAG, "removeHfLinkFolder: " + hfLinkFolder.absolutePath)
        hfLinkFolder.delete()

        val msLinkFolder = this.getMsModelPath(modelId)
        Log.d(TAG, "removeMsLinkFolder: " + msLinkFolder.absolutePath)
        msLinkFolder.delete()
        clearMmapCache(modelId)

        if (downloadListener != null) {
            val downloadInfo = getDownloadInfo(modelId)
            downloadInfo.downlodaState = RepoDownloadSate.NOT_START
            downloadListener!!.onDownloadFileRemoved(modelId)
        }
    }

    val unfinishedDownloadsSize: Int
        get() {
            var count = 0
            for (key in downloadInfoMap.keys) {
                val downloadInfo =
                    checkNotNull(downloadInfoMap[key])
                if (downloadInfo.downlodaState == RepoDownloadSate.FAILED || downloadInfo.downlodaState == RepoDownloadSate.PAUSED) {
                    count++
                }
            }
            return count
        }

    fun resumeAllDownloads() {
        for (key in downloadInfoMap.keys) {
            val downloadInfo = downloadInfoMap[key]
            if (downloadInfo!!.downlodaState == RepoDownloadSate.FAILED || downloadInfo.downlodaState == RepoDownloadSate.PAUSED) {
                startDownload(key)
            }
        }
    }

    fun pauseAllDownloads() {
        for (key in downloadInfoMap.keys) {
            val downloadInfo = downloadInfoMap[key]
            if (downloadInfo!!.isDownloadingOrPreparing()) {
                pauseDownload(key)
            }
        }
    }

    fun startForegroundService() {
        if (!foregroundServiceStarted && activeDownloadCount.get() > 0) {
            ApplicationProvider.get().startForegroundService(foregroundSerivceIntent)
        }
    }

    companion object {
        const val TAG: String = "ModelDownloadManager"
        const val REQUEST_CODE_POST_NOTIFICATIONS: Int = 998
    }
}
