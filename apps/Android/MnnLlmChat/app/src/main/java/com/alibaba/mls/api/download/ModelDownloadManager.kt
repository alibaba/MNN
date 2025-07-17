// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mls.api.download

import android.Manifest
import android.annotation.SuppressLint
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
import com.alibaba.mnnllm.android.chat.model.ChatDataManager
import com.alibaba.mnnllm.android.model.ModelUtils
import com.alibaba.mnnllm.android.modelmarket.ModelMarketItem
import com.alibaba.mnnllm.android.modelmarket.SourceSelectionDialogFragment
import com.alibaba.mnnllm.android.modelsettings.ModelConfig
import com.alibaba.mnnllm.android.utils.CurrentActivityTracker
import com.alibaba.mnnllm.android.utils.FileUtils
import com.alibaba.mnnllm.android.utils.MmapUtils.clearMmapCache
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.MainScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.atomic.AtomicInteger
import kotlin.collections.set
import kotlin.concurrent.Volatile
import com.alibaba.mnnllm.android.modelmarket.ModelRepository
import kotlinx.coroutines.runBlocking

class LoggingDownloadListener : DownloadListener {
    override fun onDownloadStart(modelId: String) {
        Log.d(ModelDownloadManager.TAG, "onDownloadStart: $modelId")
    }

    override fun onDownloadProgress(modelId: String, downloadInfo: DownloadInfo) {
        Log.v(ModelDownloadManager.TAG, "onDownloadProgress: $modelId, progress: ${downloadInfo.progress}, state: ${downloadInfo.downloadState}, speed: ${downloadInfo.speedInfo}")
    }

    override fun onDownloadFinished(modelId: String, path: String) {
        Log.d(ModelDownloadManager.TAG, "onDownloadFinished: $modelId, path: $path")
    }

    override fun onDownloadFailed(modelId: String, e: Exception) {
        Log.e(ModelDownloadManager.TAG, "onDownloadFailed: $modelId", e)
    }

    override fun onDownloadPaused(modelId: String) {
        Log.d(ModelDownloadManager.TAG, "onDownloadPaused: $modelId")
    }

    override fun onDownloadFileRemoved(modelId: String) {
        Log.d(ModelDownloadManager.TAG, "onDownloadFileRemoved: $modelId")
    }

    override fun onDownloadTotalSize(modelId: String, totalSize: Long) {
        Log.d(ModelDownloadManager.TAG, "onDownloadTotalSize: $modelId, totalSize: $totalSize")
    }
}

class ModelDownloadManager private constructor(context: Context) {
    private val listeners = CopyOnWriteArrayList<DownloadListener>()
    private val appContext = context.applicationContext
    private val cachePath = appContext.filesDir.absolutePath + "/.mnnmodels"
    private var hfDownloader: HfModelDownloader
    private var msDownloader: MsModelDownloader
    private val mlDownloader: MLModelDownloader
    private val downloadInfoMap = ConcurrentHashMap<String, DownloadInfo>()
    private val foregroundServiceIntent =
        Intent(appContext, DownloadForegroundService::class.java)
    private val activeDownloadCount =
        AtomicInteger(0)

    private var foregroundServiceStarted = false
    private var disableForegroundService  = false

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
                modelId:String,
                stage: String,
                currentFile: String?,
                saved: Long,
                total: Long
            ) {
                updateDownloadingProgress(
                    modelId, stage, currentFile,
                    saved, total
                )
            }
        }
        mlDownloader = MLModelDownloader(downloadCallback, cachePath)
        hfDownloader = HfModelDownloader(downloadCallback, cachePath)
        msDownloader = MsModelDownloader(downloadCallback, cachePath)
        addListener(LoggingDownloadListener())
    }

    fun addListener(listener: DownloadListener) {
        if (!listeners.contains(listener)) {
            Log.d(TAG, "[addListener] Adding listener: ${listener.javaClass.name}")
            listeners.add(listener)
        } else {
            Log.w(TAG, "[addListener] Listener already exists: ${listener.javaClass.name}")
        }
    }

    fun removeListener(listener: DownloadListener) {
        Log.d(TAG, "[removeListener] Removing listener: ${listener.javaClass.name}")
        listeners.remove(listener)
    }

    private fun getDownloaderForSource(source: String): ModelRepoDownloader {
        return when (source) {
            SourceSelectionDialogFragment.SOURCE_HUGGINGFACE -> hfDownloader
            SourceSelectionDialogFragment.SOURCE_MODELERS -> mlDownloader
            else -> msDownloader
        }
    }

    private fun getDownloaderForSource(source: ModelSources.ModelSourceType): ModelRepoDownloader {
        return when (source) {
            ModelSources.ModelSourceType.HUGGING_FACE -> hfDownloader
            ModelSources.ModelSourceType.MODELERS -> mlDownloader
            else -> msDownloader
        }
    }

    fun getDownloadedFile(item: ModelMarketItem): File? {
        return getDownloadedFile(item.modelId)
    }

    fun getDownloadedFile(modelId: String): File? {
        val file = getDownloaderForSource(ModelUtils.getSource(modelId)!!).getDownloadPath(modelId)
        Log.d(TAG, "getDownloadedFile: $modelId file: $file")
        if (file.exists()) {
            return file
        } else {
            return  null
        }
    }

    fun pauseDownload(modelId: String) {
        Log.d(TAG, "pauseDownload: $modelId downloadInfo : ${getDownloadInfo(modelId)}")
        if (getDownloadInfo(modelId).downloadState != DownloadState.DOWNLOADING) {
            Log.w(TAG, "not downloading")
            return
        }
        val source = ModelUtils.getSource(modelId)!!
        getDownloaderForSource(source).pause(modelId)
    }

    fun startDownload(item: ModelMarketItem) {
        Log.d(TAG, "startDownload: $item currentSource: ${item.currentSource} currentRepoPath: ${item.currentRepoPath}")
        val modelName = item.modelName
        val repoPath = item.currentRepoPath
        val source = item.currentSource
        if (repoPath.isEmpty() || source.isEmpty()) {
            setDownloadFailed(item.modelId, Exception("currentRepoPath or currentSource is empty for $modelName"))
            return
        }
        startDownload(item.modelId, source)
    }

    fun startDownload(modelId: String) {
        val splits = ModelUtils.splitSource(modelId)
        if (splits.size != 2) {
            setDownloadFailed(modelId, Exception("modelId is not in format of source:repoPath"))
            return
        }
        startDownload(modelId, splits[0])
    }

    private fun startDownload(modelId: String, source:String) {
        val downloader = getDownloaderForSource(source)
        Log.d(TAG, "startDownload: $modelId source: $source downloader: ${downloader.javaClass.name}")
        listeners.forEach { it.onDownloadStart(modelId) }
        this.updateDownloadingProgress(modelId, "Preparing", null,
            getRealDownloadSize(modelId),
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
            if (speedBytesPerSecond < 0) { //may changed the download provider
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

    fun getDownloadInfo(item: ModelMarketItem): DownloadInfo {
        return getDownloadInfo(item.modelId)
    }

    fun getDownloadInfo(modelId: String): DownloadInfo {
        Log.d(TAG, "getDownloadInfo: $modelId totalSize: ${getDownloadSizeTotal(ApplicationProvider.get(), modelId)}" +
                " progress: ${getRealDownloadSize(modelId)}")
        val source = ModelUtils.getSource(modelId)!!
        if (!downloadInfoMap.containsKey(modelId)) {
            val downloadInfo = DownloadInfo()
            if (getDownloadedFile(modelId) != null) {
                downloadInfo.downloadState = DownloadState.COMPLETED
                downloadInfo.progress = 1.0
                downloadInfo.totalSize = getDownloadSizeTotal(ApplicationProvider.get(), modelId)
            } else if (getDownloadSizeTotal(ApplicationProvider.get(), modelId) > 0) {
                val totalSize = getDownloadSizeTotal(ApplicationProvider.get(), modelId)
                val savedSize = getRealDownloadSize(modelId)
                downloadInfo.totalSize = totalSize
                downloadInfo.savedSize = savedSize
                downloadInfo.progress = savedSize.toDouble() / totalSize
                downloadInfo.downloadState = if (savedSize > 0) DownloadState.PAUSED else DownloadState.NOT_START
            } else {
                downloadInfo.downloadState = DownloadState.NOT_START
                downloadInfo.progress = 0.0
                getRepoSize(modelId, source)
            }
            downloadInfoMap[modelId] = downloadInfo
        }
        return downloadInfoMap[modelId]!!
    }

    /**
     * Get the real download size for a model.
     * If getDownloadSizeSaved returns a value less than 0, 
     * it will get the actual file size using FileUtils.
     */
    fun getRealDownloadSize(modelId: String): Long {
        val savedSize = getDownloadSizeSaved(ApplicationProvider.get(), modelId)
        if (savedSize < 0) {
            val splits = ModelUtils.splitSource(modelId)
            val source = splits[0]
            val downloader = getDownloaderForSource(source)
            val realFile = downloader.repoModelRealFile(modelId)
            Log.d(TAG, "getRealDownloadSize realFile: ${realFile.absolutePath}")
            val fileSize = FileUtils.getFileSize(realFile)
            saveDownloadSizeSaved(ApplicationProvider.get(), modelId, fileSize)
            return FileUtils.getFileSize(realFile)
            return 0L
        }
        return savedSize
    }

    private fun getRepoSize(
        modelId: String,
        source: Any,
        modelName: String = modelId
    ) {
        MainScope().launch {
            val downloader = when (source) {
                is String -> getDownloaderForSource(source)
                is ModelSources.ModelSourceType -> getDownloaderForSource(source)
                else -> throw IllegalArgumentException("Unsupported source type: ${source::class.java.name}")
            }
            val repoSize = downloader.getRepoSize(modelId)
            if (repoSize > 0 && getDownloadSizeTotal(ApplicationProvider.get(), modelName) <= 0L) {
                saveDownloadSizeTotal(ApplicationProvider.get(), modelName, repoSize)
                downloadInfoMap[modelName]?.totalSize = repoSize
                listeners.forEach { it.onDownloadTotalSize(modelName, repoSize) }
            }
        }
    }

    private fun setDownloadFinished(modelId: String, path: String) {
        val downloadInfo = downloadInfoMap[modelId] ?: return
        downloadInfo.downloadState = DownloadState.COMPLETED
        // Record download history
        try {
            val modelType = runBlocking {
                val modelRepository = ModelRepository(appContext)
                modelRepository.getModelType(modelId)
            }
            ChatDataManager.getInstance(appContext)
                .recordDownloadHistory(modelId, path, modelType)
        } catch (e: Exception) {
            Log.w(TAG, "Failed to record download history for $modelId", e)
        }
        
        listeners.forEach { it.onDownloadFinished(modelId, path) }
    }

    private fun setDownloadPaused(modelId: String) {
        val info = downloadInfoMap.getOrPut(modelId) { DownloadInfo() }
        info.downloadState = DownloadState.PAUSED
        listeners.forEach { it.onDownloadPaused(modelId) }
    }

    private fun onDownloadTaskAdded(count: Int) {
        // Do not start foreground service in Google Play build
        if (disableForegroundService) {
            return
        }
        
        if (count == 1) {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
                if (ContextCompat.checkSelfPermission(
                        appContext,
                        Manifest.permission.POST_NOTIFICATIONS
                    )
                    != PackageManager.PERMISSION_GRANTED
                ) {
                    val currentActivity = CurrentActivityTracker.currentActivity
                    if (currentActivity != null) {
                        ActivityCompat.requestPermissions(
                            currentActivity,
                            arrayOf(Manifest.permission.POST_NOTIFICATIONS),
                            REQUEST_CODE_POST_NOTIFICATIONS
                        )
                    } else {
                        Log.w(TAG, "No current activity available to request permissions")
                        startForegroundService()
                    }
                } else {
                    startForegroundService()
                }
            }
        }
    }

    private fun startForegroundService() {
        // Do not start foreground service in Google Play build
        if (disableForegroundService) {
            return
        }
        
        try {
            ApplicationProvider.get().startForegroundService(foregroundServiceIntent)
            foregroundServiceStarted = true
        } catch (e: Exception) {
            Log.e(TAG, "start foreground service failed", e)
            foregroundServiceStarted = false
        }
    }

    private fun onDownloadTaskRemoved(count: Int) {
        // Do not manage foreground service in Google Play build
        if (disableForegroundService) {
            return
        }
        
        if (count == 0) {
            ApplicationProvider.get().stopService(foregroundServiceIntent)
            foregroundServiceStarted = false
        }
    }

    private fun setDownloadFailed(modelId: String, e: Exception) {
        Log.e(TAG, "[setDownloadFailed] Setting failure for $modelId. Notifying ${listeners.size} listeners.", e)
        val info = downloadInfoMap.getOrPut(modelId) { DownloadInfo() }
        info.downloadState = DownloadState.FAILED
        info.errorMessage = e.message
        listeners.forEach {
            Log.d(TAG, "[setDownloadFailed] Notifying listener: ${it.javaClass.simpleName}")
            it.onDownloadFailed(modelId, e)
        }
    }

    private fun updateDownloadingProgress(
        modelId: String,
        stage: String,
        currentFile: String?,
        savedSize: Long,
        totalSize: Long
    ) {
        if (totalSize <= 0) {
            return
        }
        val downloadInfo = downloadInfoMap.getOrPut(modelId) { DownloadInfo() }
        val currentTime = System.currentTimeMillis()
        if (stage == downloadInfo.progressStage && currentTime - downloadInfo.lastProgressUpdateTime < 1000) {
            return
        }
        downloadInfo.lastProgressUpdateTime = currentTime
        downloadInfo.downloadState = DownloadState.DOWNLOADING
        downloadInfo.progressStage = stage
        downloadInfo.currentFile = currentFile
        downloadInfo.totalSize = totalSize
        if (downloadInfo.savedSize <= 0) {
            downloadInfo.savedSize = savedSize
        }
        downloadInfo.progress = savedSize.toDouble() / totalSize
        saveDownloadSizeSaved(ApplicationProvider.get(), modelId, savedSize)
        saveDownloadSizeTotal(ApplicationProvider.get(), modelId, totalSize)
        calculateDownloadSpeed(downloadInfo, savedSize)
        Log.v(TAG, "[updateDownloadingProgress] Notifying ${listeners.size} listeners for $modelId")
        listeners.forEach { it.onDownloadProgress(modelId, downloadInfo) }
    }

    suspend fun deleteModel(item: ModelMarketItem) {
        deleteModel(item.modelId, item.currentSource, item.currentRepoPath)
    }

    suspend fun deleteModel(modelId:String, modelSource:String, modelPath:String) {
        withContext(Dispatchers.IO) {
            getDownloaderForSource(modelSource).deleteRepo(modelId)
            removeProgress(ApplicationProvider.get(), modelId)
            clearMmapCache(modelId)
            ModelConfig.getModelConfigDir(modelId).let {
                DownloadFileUtils.deleteDirectoryRecursively(File(it))
            }
        }
        val downloadInfo = getDownloadInfo(modelId)
        downloadInfo.downloadState = DownloadState.NOT_START
        listeners.forEach { it.onDownloadFileRemoved(modelId) }
    }

    suspend fun deleteModel(modelId: String) {
        val splits = ModelUtils.splitSource(modelId)
        if (splits.size == 2 ) {
            deleteModel(modelId, splits[0], splits[1])
        }
    }

    val unfinishedDownloadsSize: Int
        get() {
            var count = 0
            for (key in downloadInfoMap.keys) {
                val downloadInfo =
                    checkNotNull(downloadInfoMap[key])
                if (downloadInfo.downloadState == DownloadState.FAILED || downloadInfo.downloadState == DownloadState.PAUSED) {
                    count++
                }
            }
            return count
        }

    fun resumeAllDownloads() {
        for (key in downloadInfoMap.keys) {
            val downloadInfo = downloadInfoMap[key]
            if (downloadInfo!!.downloadState == DownloadState.FAILED
                || downloadInfo.downloadState == DownloadState.PAUSED) {
                startDownload(key)
            }
        }
    }

    fun pauseAllDownloads() {
        for (key in downloadInfoMap.keys) {
             val downloadInfo = downloadInfoMap[key]
            if (downloadInfo!!.downloadState == DownloadState.DOWNLOADING) {
                pauseDownload(key)
            }
        }
    }

    fun tryStartForegroundService() {
        // Do not start foreground service in Google Play build
        if (disableForegroundService) {
            return
        }
        
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
                        instance = ModelDownloadManager(context.applicationContext)
                    }
                }
            }
            return instance!!
        }
    }
}