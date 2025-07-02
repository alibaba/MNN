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

import com.alibaba.mls.api.download.DownloadPersistentData.getDownloadSizeSaved
import com.alibaba.mls.api.download.DownloadPersistentData.getDownloadSizeTotal
import com.alibaba.mls.api.download.DownloadPersistentData.removeProgress
import com.alibaba.mls.api.download.DownloadPersistentData.saveDownloadSizeSaved
import com.alibaba.mls.api.download.DownloadPersistentData.saveDownloadSizeTotal
import com.alibaba.mls.api.download.hf.HfModelDownloader
import com.alibaba.mls.api.download.ml.MLModelDownloader
import com.alibaba.mls.api.download.ms.MsModelDownloader
import com.alibaba.mls.api.source.ModelSources
import com.alibaba.mnnllm.android.modelsettings.ModelConfig
import com.alibaba.mnnllm.android.utils.FileUtils
import com.alibaba.mnnllm.android.utils.FileUtils.clearMmapCache
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.MainScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.util.concurrent.atomic.AtomicInteger
import kotlin.collections.set
import kotlin.concurrent.Volatile

class ModelDownloadManager private constructor(private val context: Context) {
    private var downloadListener: DownloadListener? = null
    private val cachePath = context.filesDir.absolutePath + "/.mnnmodels"
    private lateinit var downloader: ModelRepoDownloader
    private var hfDownloader:HfModelDownloader
    private var msDownloader:MsModelDownloader
    private val mlDownloader:MLModelDownloader
    private val downloadInfoMap = HashMap<String, DownloadInfo>()
    private val foregroundServiceIntent =
        Intent(context.applicationContext, DownloadForegroundService::class.java)
    private val activeDownloadCount =
        AtomicInteger(0)

    private var foregroundServiceStarted = false

    init {
        val downloadCallback = object : ModelRepoDownloader.ModelRepoDownloadCallback {
            override fun onDownloadFailed(modelId: String, e: Exception) {
                setDownloadFailed(modelId, e)
            }

            override fun onDownloadTaskAdded() {
                onDownloadTaskAdded(activeDownloadCount.incrementAndGet())
            }

            override fun onDownloadTaskRemoved() {
                onDownloadTaskRemoved(activeDownloadCount.decrementAndGet())
            }

            override fun onDownloadPaused(modelId: String) {
                setDownloadPaused(modelId)
            }

            override fun onDownloadFileFinished(modelId: String, absolutePath: String) {
                setDownloadFinished(modelId, absolutePath)
            }

            override fun onDownloadingProgress(
                modelId: String,
                stage: String,
                currentFile: String?,
                saved: Long,
                total: Long
            ) {
                updateDownloadingProgress(
                    modelId, "file", currentFile,
                    saved, total
                )
            }
        }
        mlDownloader = MLModelDownloader(downloadCallback, cachePath)
        hfDownloader = HfModelDownloader(downloadCallback, cachePath)
        msDownloader = MsModelDownloader(downloadCallback, cachePath)
        setDownloader(ModelSources.get().remoteSourceType)
    }

    fun setListener(downloadListener: DownloadListener?) {
        this.downloadListener = downloadListener
    }

    private fun setDownloader(sourceType: ModelSources.ModelSourceType) {

        this.downloader = if (sourceType == ModelSources.ModelSourceType.HUGGING_FACE) {
            hfDownloader
        } else if (sourceType == ModelSources.ModelSourceType.MODELERS)  {
            mlDownloader
        } else {
            msDownloader
        }
    }

    fun getDownloadedFile(modelId: String): File? {
        val file = downloader.getDownloadPath(modelId)
        if (file.exists()) {
            return file
        } else if (HfModelDownloader.getModelPath(cachePath, modelId).exists()) {
            return HfModelDownloader.getModelPath(cachePath, modelId)
        } else if (MsModelDownloader.getModelPath(
                MsModelDownloader.getCachePathRoot(cachePath)
                , modelId).exists()) {
            return MsModelDownloader.getModelPath(
                MsModelDownloader.getCachePathRoot(cachePath)
                , modelId)
        } else if (MLModelDownloader.getModelPath(
                MLModelDownloader.getCachePathRoot(cachePath)
                , modelId).exists()) {
            return MLModelDownloader.getModelPath(
                MLModelDownloader.getCachePathRoot(cachePath)
                , modelId)
        }
        return null
    }

    fun pauseDownload(modelId: String) {
        if (getDownloadInfo(modelId).downlodaState != DownloadInfo.DownloadSate.DOWNLOADING) {
            return
        }
        downloader.pause(modelId)
    }


    fun startDownload(modelId: String) {
        setDownloader(ModelSources.get().remoteSourceType)
        if (downloadListener != null) {
            downloadListener!!.onDownloadStart(modelId)
        }
        this.updateDownloadingProgress(modelId, "Preparing", null,
            getDownloadSizeSaved(ApplicationProvider.get(), modelId),
            getDownloadSizeTotal(ApplicationProvider.get(), modelId))
        downloader.download(modelId)
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
            var speedBytesPerSecond = ((currentDownloadSize - lastDownloadSize) * 1000.0) / timeDiff
            if (speedBytesPerSecond < 0 ) { //may changed the download provider
                speedBytesPerSecond = 0.0
            }
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

    fun getDownloadInfo(modelId: String): DownloadInfo {
        if (!downloadInfoMap.containsKey(modelId)) {
            val downloadInfo = DownloadInfo()
            if (getDownloadedFile(modelId) != null) {
                downloadInfo.downlodaState = DownloadInfo.DownloadSate.COMPLETED
                downloadInfo.progress = 1.0
            } else if (getDownloadSizeTotal(ApplicationProvider.get(), modelId) > 0) {
                val totalSize = getDownloadSizeTotal(ApplicationProvider.get(), modelId)
                val savedSize = getDownloadSizeSaved(ApplicationProvider.get(), modelId)
                downloadInfo.totalSize = totalSize
                downloadInfo.savedSize = savedSize
                downloadInfo.progress = savedSize.toDouble() / totalSize
                downloadInfo.downlodaState = if (savedSize > 0) DownloadInfo.DownloadSate.PAUSED else DownloadInfo.DownloadSate.NOT_START
            } else {
                downloadInfo.downlodaState = DownloadInfo.DownloadSate.NOT_START
                downloadInfo.progress = 0.0
                getRepoSize(modelId)
            }
            downloadInfoMap[modelId] = downloadInfo
        }
        return downloadInfoMap[modelId]!!
    }

    private fun getRepoSize(modelId: String) {
        MainScope().launch {
            val repoSize = downloader.getRepoSize(modelId)
            if (repoSize > 0 && getDownloadSizeTotal(ApplicationProvider.get(), modelId) <= 0L) {
                saveDownloadSizeTotal(ApplicationProvider.get(), modelId, repoSize)
                getDownloadInfo(modelId).totalSize = repoSize
                downloadListener?.onDownloadTotalSize(modelId, repoSize)
            }
        }
    }

    private fun setDownloadFinished(modelId: String, path: String) {
        val downloadInfo = getDownloadInfo(modelId)
        downloadInfo.downlodaState = DownloadInfo.DownloadSate.COMPLETED
        if (downloadListener != null) {
            downloadListener!!.onDownloadFinished(modelId, path)
        }
    }

    private fun setDownloadPaused(modelId: String) {
        val downloadInfo = getDownloadInfo(modelId)
        downloadInfo.downlodaState = DownloadInfo.DownloadSate.PAUSED
        if (downloadListener != null) {
            downloadListener!!.onDownloadPaused(modelId)
        }
    }

    private fun onDownloadTaskAdded(count: Int) {
        if (count == 1) {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
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
                    startForegroundService()
                }
            }
        }
    }

    private fun startForegroundService() {
        try {
            ApplicationProvider.get().startForegroundService(foregroundServiceIntent)
            foregroundServiceStarted = true
        } catch (e: Exception) {
            Log.e(TAG, "start foreground service failed", e)
            foregroundServiceStarted = false
        }
    }

    private fun onDownloadTaskRemoved(count: Int) {
        if (count == 0) {
            ApplicationProvider.get().stopService(foregroundServiceIntent)
            foregroundServiceStarted = false
        }
    }

    fun setDownloadFailed(modelId: String, e: Exception) {
        Log.e(TAG, "onDownloadFailed: $modelId", e)
        val downloadInfo = getDownloadInfo(modelId)
        downloadInfo.downlodaState = DownloadInfo.DownloadSate.FAILED
        downloadInfo.errorMessage = e.message
        if (downloadListener != null) {
            downloadListener!!.onDownloadFailed(modelId, e)
        }
    }

    private fun updateDownloadingProgress(
        modelId: String,
        stage: String,
        currentFile: String?,
        saved: Long,
        total: Long
    ) {
        if (!downloadInfoMap.containsKey(modelId)) {
            val downloadInfo = DownloadInfo()
            downloadInfoMap[modelId] = downloadInfo
        }
        val downloadInfo = checkNotNull(downloadInfoMap[modelId])
        downloadInfo.progress = if (total > 0) (saved.toDouble() / total) else 0.0
        calculateDownloadSpeed(downloadInfo, saved)
        downloadInfo.totalSize = total
        downloadInfo.progressStage = stage
        downloadInfo.currentFile = currentFile
        downloadInfo.downlodaState = DownloadInfo.DownloadSate.DOWNLOADING
        saveDownloadSizeTotal(ApplicationProvider.get(), modelId, total)
        saveDownloadSizeSaved(ApplicationProvider.get(), modelId, saved)
        if (downloadListener != null) {
            downloadListener!!.onDownloadProgress(modelId, downloadInfo)
        }
    }

    suspend fun deleteModel(modelId: String) {
        withContext(Dispatchers.IO) {
            msDownloader.deleteRepo(modelId)
            hfDownloader.deleteRepo(modelId)
            mlDownloader.deleteRepo(modelId)
            removeProgress(ApplicationProvider.get(), modelId)
            clearMmapCache(modelId)
            ModelConfig.getModelConfigDir(modelId).let {
                DownloadFileUtils.deleteDirectoryRecursively(File(it))
            }
        }
        if (downloadListener != null) {
            val downloadInfo = getDownloadInfo(modelId)
            downloadInfo.downlodaState = DownloadInfo.DownloadSate.NOT_START
            downloadListener!!.onDownloadFileRemoved(modelId)
        }
    }

    val unfinishedDownloadsSize: Int
        get() {
            var count = 0
            for (key in downloadInfoMap.keys) {
                val downloadInfo =
                    checkNotNull(downloadInfoMap[key])
                if (downloadInfo.downlodaState == DownloadInfo.DownloadSate.FAILED || downloadInfo.downlodaState == DownloadInfo.DownloadSate.PAUSED) {
                    count++
                }
            }
            return count
        }

    fun resumeAllDownloads() {
        for (key in downloadInfoMap.keys) {
            val downloadInfo = downloadInfoMap[key]
            if (downloadInfo!!.downlodaState == DownloadInfo.DownloadSate.FAILED
                || downloadInfo.downlodaState == DownloadInfo.DownloadSate.PAUSED) {
                startDownload(key)
            }
        }
    }

    fun pauseAllDownloads() {
        for (key in downloadInfoMap.keys) {
            val downloadInfo = downloadInfoMap[key]
            if (downloadInfo!!.downlodaState == DownloadInfo.DownloadSate.DOWNLOADING) {
                pauseDownload(key)
            }
        }
    }

    fun tryStartForegroundService() {
        if (!foregroundServiceStarted && activeDownloadCount.get() > 0) {
            startForegroundService()
        }
    }

    companion object {
        @Volatile
        private var instance: ModelDownloadManager? = null
        const val TAG: String = "ModelDownloadManager"

        const val REQUEST_CODE_POST_NOTIFICATIONS: Int = 998

        fun getInstance(context: Context): ModelDownloadManager {
            if (instance == null) {
                synchronized(ModelDownloadManager::class.java) {
                    if (instance == null) {
                        instance = ModelDownloadManager(context)
                    }
                }
            }
            return instance!!
        }
    }
}
