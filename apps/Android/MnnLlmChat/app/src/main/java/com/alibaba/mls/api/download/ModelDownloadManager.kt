// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mls.api.download

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.alibaba.mls.api.ApplicationProvider
import com.alibaba.mls.api.download.hf.HfModelDownloader
import com.alibaba.mls.api.download.ml.MLModelDownloader
import com.alibaba.mls.api.download.ms.MsModelDownloader
import com.alibaba.mls.api.source.ModelSources
import com.alibaba.mnnllm.android.chat.model.ChatDataManager
import com.alibaba.mnnllm.android.model.ModelTypeUtils
import com.alibaba.mnnllm.android.model.ModelUtils
import com.alibaba.mnnllm.android.modelmarket.ModelMarketItem
import com.alibaba.mnnllm.android.modelmarket.ModelRepository
import com.alibaba.mnnllm.android.modelsettings.ModelConfig
import com.alibaba.mnnllm.android.qnn.QnnModule
import com.alibaba.mnnllm.android.utils.CurrentActivityTracker
import com.alibaba.mnnllm.android.utils.FileUtils
import com.alibaba.mnnllm.android.utils.MmapUtils.clearMmapCache
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import timber.log.Timber
import java.io.File
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.atomic.AtomicInteger
import kotlin.collections.set
import kotlin.concurrent.Volatile

/**
 * Specialized listener for QNN library downloads that uses coroutines to wait for completion
 */
class QnnDownloadListener(
    private val modelId: String,
    private val completionChannel: Channel<DownloadResult>
) : DownloadListener {
    
    override fun onDownloadStart(modelId: String) {
        Timber.tag(ModelDownloadManager.TAG).d("[QNNDownload] Download started: $modelId")
    }

    override fun onDownloadProgress(modelId: String, downloadInfo: DownloadInfo) {
        Timber.tag(ModelDownloadManager.TAG).v("[QNNDownload] Progress: $modelId, progress: ${downloadInfo.progress}, state: ${downloadInfo.downloadState}")
    }

    override fun onDownloadFinished(modelId: String, path: String) {
        Timber.tag(ModelDownloadManager.TAG).d("[QNNDownload] Download finished: $modelId, path: $path")
        if (modelId == this.modelId) {
            completionChannel.trySend(DownloadResult.Success(path))
        }
    }

    override fun onDownloadFailed(modelId: String, e: Exception) {
        Timber.tag(ModelDownloadManager.TAG).e(e, "[QNNDownload] Download failed: $modelId")
        if (modelId == this.modelId) {
            completionChannel.trySend(DownloadResult.Failure(e))
        }
    }

    override fun onDownloadPaused(modelId: String) {
        Timber.tag(ModelDownloadManager.TAG).d("[QNNDownload] Download paused: $modelId")
    }

    override fun onDownloadFileRemoved(modelId: String) {
        Timber.tag(ModelDownloadManager.TAG).d("[QNNDownload] Download file removed: $modelId")
    }

    override fun onDownloadTotalSize(modelId: String, totalSize: Long) {
        Timber.tag(ModelDownloadManager.TAG).d("[QNNDownload] Total size: $modelId, totalSize: $totalSize")
    }

    override fun onDownloadHasUpdate(modelId: String, downloadInfo: DownloadInfo) {
        Timber.tag(ModelDownloadManager.TAG).d("[QNNDownload] Has update: $modelId, uploadTime: ${downloadInfo.remoteUpdateTime} downloadTime: ${downloadInfo.downloadedTime}")
    }
}

/**
 * Result of QNN download operation
 */
sealed class DownloadResult {
    data class Success(val path: String) : DownloadResult()
    data class Failure(val exception: Exception) : DownloadResult()
}

class LoggingDownloadListener : DownloadListener {
    override fun onDownloadStart(modelId: String) {
        Timber.tag(ModelDownloadManager.TAG).d("onDownloadStart: $modelId")
    }

    override fun onDownloadProgress(modelId: String, downloadInfo: DownloadInfo) {
        Timber.tag(ModelDownloadManager.TAG).v("onDownloadProgress: $modelId, progress: ${downloadInfo.progress}, state: ${downloadInfo.downloadState}, speed: ${downloadInfo.speedInfo} stage: ${downloadInfo.progressStage}")
    }

    override fun onDownloadFinished(modelId: String, path: String) {
        Timber.tag(ModelDownloadManager.TAG).d("onDownloadFinished: $modelId, path: $path")
    }

    override fun onDownloadFailed(modelId: String, e: Exception) {
        Timber.tag(ModelDownloadManager.TAG).e(e, "onDownloadFailed: $modelId")
    }

    override fun onDownloadPaused(modelId: String) {
        Timber.tag(ModelDownloadManager.TAG).d("onDownloadPaused: $modelId")
    }

    override fun onDownloadFileRemoved(modelId: String) {
        Timber.tag(ModelDownloadManager.TAG).d("onDownloadFileRemoved: $modelId")
    }

    override fun onDownloadTotalSize(modelId: String, totalSize: Long) {
        Timber.tag(ModelDownloadManager.TAG).d("onDownloadTotalSize: $modelId, totalSize: $totalSize")
    }

    override fun onDownloadHasUpdate(modelId: String, downloadInfo: DownloadInfo) {
        Timber.tag(ModelDownloadManager.TAG).d("modelId:$modelId onDownloadTotalSize: $modelId, uploadTime: ${downloadInfo.remoteUpdateTime} downloadTime: ${downloadInfo.downloadedTime}")
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
    private val checkedForUpdateModelIds = ConcurrentHashMap<String, Boolean>()
    private val activeDownloadModelNames = ConcurrentHashMap<String, String>()

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

            override fun onRepoInfo(modelId: String, lastModified: Long, repoSize: Long) {
                onRepoInfoReceived(modelId, lastModified, repoSize)
            }
        }
        mlDownloader = MLModelDownloader(downloadCallback, cachePath)
        hfDownloader = HfModelDownloader(downloadCallback, cachePath)
        msDownloader = MsModelDownloader(downloadCallback, cachePath)
        addListener(LoggingDownloadListener())
    }

    fun addListener(listener: DownloadListener) {
        if (!listeners.contains(listener)) {
            Timber.tag(TAG).d("[addListener] Adding listener: ${listener.javaClass.name}")
            listeners.add(listener)
        } else {
            Timber.tag(TAG).w("[addListener] Listener already exists: ${listener.javaClass.name}")
        }
    }

    fun removeListener(listener: DownloadListener) {
        Timber.tag(TAG).d("[removeListener] Removing listener: ${listener.javaClass.name}")
        listeners.remove(listener)
    }

    private fun getDownloaderForSource(source: String): ModelRepoDownloader {
        return when (source) {
            ModelSources.sourceHuffingFace-> hfDownloader
            ModelSources.sourceModelers -> mlDownloader
            else -> msDownloader
        }
    }

    fun getDownloadedFile(modelId: String): File? {
        val file = getDownloaderForSource(ModelUtils.getSource(modelId)!!).getDownloadPath(modelId)
        Timber.tag(TAG).d("getDownloadedFile: $modelId file: $file")
        if (file.exists()) {
            return file
        } else {
            return  null
        }
    }

    fun pauseDownload(modelId: String) {
        Timber.tag(TAG).d("pauseDownload: $modelId downloadInfo : ${getDownloadInfo(modelId)}")
        if (getDownloadInfo(modelId).downloadState != DownloadState.DOWNLOADING) {
            Timber.tag(TAG).w("not downloading")
            return
        }
        val source = ModelUtils.getSource(modelId)!!
        getDownloaderForSource(source).pause(modelId)
    }

    suspend fun downloadQnnLibs(): Boolean {
        if (!QnnModule.deviceSupported()) {
            return false
        }
        
        if (QnnModule.isQnnLibsDownloaded(appContext)) {
            Timber.tag(TAG).d("[QNNDownload] QNN libs already downloaded")
            return true
        }
        
        Timber.tag(TAG).d("[QNNDownload] QNN libs not copied yet, starting prerequisite download")
        return withContext(DownloadCoroutineManager.downloadDispatcher) {
            try {
                Timber.tag(TAG).d("[QNNDownload] Starting QNN ARM64 libraries download")
                val libs = ModelRepository.getMarketDataSuspend().libs
                // Find qnn_arm64_libs
                val qnnLibsItem = libs.find { it.modelName.equals("qnn_arm64_libs", ignoreCase = true) }
                if (qnnLibsItem == null) {
                    Timber.tag(TAG).e("[QNNDownload] qnn_arm64_libs not found in libs list")
                    return@withContext false
                }

                Timber.tag(TAG).d("[QNNDownload] Found qnn_arm64_libs item: ${qnnLibsItem.modelId}")

                // Create completion channel and listener for this specific download
                val completionChannel = Channel<DownloadResult>(Channel.CONFLATED)
                val qnnListener = QnnDownloadListener(qnnLibsItem.modelId, completionChannel)
                
                // Add the listener temporarily
                addListener(qnnListener)

                try {
                    // Start the download
                    val downloader = getDownloaderForSource(qnnLibsItem.currentSource)
                    downloader.download(qnnLibsItem.modelId)

                    // Wait for download completion using the listener
                    Timber.tag(TAG).d("[QNNDownload] Waiting for download completion...")
                    val result = completionChannel.receive()
                    
                    when (result) {
                        is DownloadResult.Success -> {
                            Timber.tag(TAG).d("[QNNDownload] Download completed successfully: ${result.path}")
                            
                            // Get the downloaded file to verify
                            val downloadedFile = getDownloadedFile(qnnLibsItem.modelId)
                            if (downloadedFile == null || !downloadedFile.exists()) {
                                Timber.tag(TAG).e("[QNNDownload] QNN libs downloaded file not found")
                                return@withContext false
                            }

                            // Mark as downloaded with path
                            QnnModule.markQnnLibsDownloaded(appContext, downloadedFile.absolutePath)
                            Timber.tag(TAG).d("[QNNDownload] QNN libs marked as downloaded at ${downloadedFile.absolutePath}")
                            return@withContext true
                        }
                        is DownloadResult.Failure -> {
                            Timber.tag(TAG).e(result.exception, "[QNNDownload] QNN libs download failed")
                            return@withContext false
                        }
                    }
                } finally {
                    // Always remove the temporary listener
                    removeListener(qnnListener)
                }
            } catch (e: Exception) {
                Timber.tag(TAG).e(e, "[QNNDownload] Error downloading QNN libs")
                return@withContext false
            }
        }
    }

    fun startDownload(item: ModelMarketItem) {
        Timber.tag(TAG).d("startDownload: $item currentSource: ${item.currentSource} currentRepoPath: ${item.currentRepoPath}")
        val modelName = item.modelName
        val repoPath = item.currentRepoPath
        val source = item.currentSource
        if (repoPath.isEmpty() || source.isEmpty()) {
            setDownloadFailed(item.modelId, Exception("currentRepoPath or currentSource is empty for $modelName"))
            return
        }
        val isQnnModel = ModelTypeUtils.isQnnModel(item.tags)
        val isDeviceSupportQnn = QnnModule.deviceSupported()
        if (isQnnModel) {
            Timber.tag(TAG).d("[QNNDownload] Model tagged as QNN: $modelName isDeviceSupportQnn: $isDeviceSupportQnn")
        }
        
        if (isQnnModel && isDeviceSupportQnn) {
            DownloadCoroutineManager.launchDownload {
                try {
                    // Handle QNN libs download
                    val success = downloadQnnLibs()
                    Timber.tag(TAG).d("[QNNDownload] QNN libs download ${if (success) "succeeded" else "failed"}")
                } catch (e: Exception) {
                    Timber.tag(TAG).e(e, "[QNNDownload] Error in QNN libs download")
                }
                startDownload(item.modelId, source, modelName)
            }
        } else {
            startDownload(item.modelId, source, modelName)
        }
    }

    fun startDownload(modelId: String) {
        val splits = ModelUtils.splitSource(modelId)
        if (splits.size != 2) {
            setDownloadFailed(modelId, Exception("modelId is not in format of source:repoPath"))
            return
        }
        startDownload(modelId, splits[0])
    }

    private fun startDownload(modelId: String, source:String, modelName: String? = null) {
        val downloader = getDownloaderForSource(source)
        Timber.tag(TAG).d("startDownload: $modelId source: $source downloader: ${downloader.javaClass.name}")
        listeners.forEach { it.onDownloadStart(modelId) }
        this.updateDownloadingProgress(modelId, "Preparing", null,
            getRealDownloadSize(modelId),
            DownloadPersistentData.getDownloadSizeTotal(ApplicationProvider.get(), modelId))
        val displayName = modelName ?: ModelUtils.getModelName(modelId) ?: modelId
        addActiveDownload(modelId, displayName)
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
        Timber.tag(TAG).d("getDownloadInfo: $modelId totalSize: ${DownloadPersistentData.getDownloadSizeTotal(ApplicationProvider.get(), modelId)}" +
                " progress: ${getRealDownloadSize(modelId)}")
        if (!downloadInfoMap.containsKey(modelId)) {
            val downloadInfo = DownloadInfo()
            if (getDownloadedFile(modelId) != null) {
                downloadInfo.downloadState = DownloadState.COMPLETED
                downloadInfo.progress = 1.0
                downloadInfo.totalSize = DownloadPersistentData.getDownloadSizeTotal(ApplicationProvider.get(), modelId)
                downloadInfo.downloadedTime = DownloadPersistentData.getDownloadedTime(ApplicationProvider.get(), modelId)
                if (downloadInfo.downloadedTime <= 0) {
                    downloadInfo.downloadedTime = ChatDataManager.getInstance(appContext).getDownloadTime(modelId) / 1000
                }
            } else if (DownloadPersistentData.getDownloadSizeTotal(ApplicationProvider.get(), modelId) > 0) {
                val totalSize = DownloadPersistentData.getDownloadSizeTotal(ApplicationProvider.get(), modelId)
                val savedSize = getRealDownloadSize(modelId)
                downloadInfo.totalSize = totalSize
                downloadInfo.savedSize = savedSize
                downloadInfo.progress = savedSize.toDouble() / totalSize
                downloadInfo.downloadState = if (savedSize > 0) DownloadState.PAUSED else DownloadState.NOT_START
            } else {
                downloadInfo.downloadState = DownloadState.NOT_START
                downloadInfo.progress = 0.0
            }
            downloadInfoMap[modelId] = downloadInfo
            if (downloadInfo.totalSize < 100) {
                val marketSize = DownloadPersistentData.getMarketSizeTotal(ApplicationProvider.get(), modelId)
                Timber.tag(TAG).d("getMarketSize for ${modelId} size: $marketSize")
                if (marketSize > 0) {
                    downloadInfo.totalSize = marketSize
                }
            }
        }
        val downloadInfo = downloadInfoMap[modelId]!!
        return downloadInfo
    }

    /**
     * Get the real download size for a model.
     * If getDownloadSizeSaved returns a value less than 0, 
     * it will get the actual file size using FileUtils.
     */
    fun getRealDownloadSize(modelId: String): Long {
        val savedSize = DownloadPersistentData.getDownloadSizeSaved(ApplicationProvider.get(), modelId)
        if (savedSize < 0) {
            val splits = ModelUtils.splitSource(modelId)
            val source = splits[0]
            val downloader = getDownloaderForSource(source)
            val realFile = downloader.repoModelRealFile(modelId)
            Timber.tag(TAG).d("getRealDownloadSize realFile: ${realFile.absolutePath}")
            val fileSize = FileUtils.getFileSize(realFile)
            DownloadPersistentData.saveDownloadSizeSaved(ApplicationProvider.get(), modelId, fileSize)
            return FileUtils.getFileSize(realFile)
            return 0L
        }
        return savedSize
    }

    private fun setDownloadFinished(modelId: String, path: String) {
        val downloadInfo = downloadInfoMap[modelId] ?: return
        downloadInfo.downloadState = DownloadState.COMPLETED
        downloadInfo.progress = 1.0
        
        // Set downloadedTime to remoteUpdateTime and save to DataStore
        if (downloadInfo.remoteUpdateTime > 0) {
            downloadInfo.downloadedTime = downloadInfo.remoteUpdateTime
            DownloadPersistentData.saveDownloadedTime(ApplicationProvider.get(), modelId, downloadInfo.downloadedTime)
        }
        downloadInfo.hasUpdate = false
        
        // Record download history
        try {
            val modelType = runBlocking {
                ModelRepository.getModelTypeSuspend(modelId)
            }
            ChatDataManager.getInstance(appContext)
                .recordDownloadHistory(modelId, path, modelType)
        } catch (e: Exception) {
            Timber.tag(TAG).w(e, "Failed to record download history for $modelId")
        }
        
        listeners.forEach { it.onDownloadFinished(modelId, path) }
        removeActiveDownload(modelId)
    }

    private fun setDownloadPaused(modelId: String) {
        val info = downloadInfoMap.getOrPut(modelId) { DownloadInfo() }
        info.downloadState = DownloadState.PAUSED
        removeActiveDownload(modelId)
        listeners.forEach { it.onDownloadPaused(modelId) }
    }

    private fun onDownloadTaskAdded(count: Int) {
        // Do not start foreground service in Google Play build
        if (disableForegroundService) {
            return
        }
        
        if (count == 1) {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
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
                        Timber.tag(TAG).w("No current activity available to request permissions")
                        startForegroundService()
                    }
                } else {
                    startForegroundService()
                }
            } else {
                // For Android 13 and below, start foreground service directly
                startForegroundService()
            }
        }
        updateNotification()
    }

    private fun startForegroundService() {
        // Do not start foreground service in Google Play build
        if (disableForegroundService) {
            Timber.tag(TAG).d("startForegroundService: skipped - disableForegroundService is true")
            return
        }
        
        try {
            // Add download count and model name to intent
            val count = activeDownloadCount.get()
            val modelName = getActiveDownloadModelName()
            foregroundServiceIntent.putExtra(DownloadForegroundService.EXTRA_DOWNLOAD_COUNT, count)
            foregroundServiceIntent.putExtra(DownloadForegroundService.EXTRA_MODEL_NAME, modelName)
            
            Timber.tag(TAG).d("startForegroundService: starting service with count=$count, modelName=$modelName")
            ApplicationProvider.get().startForegroundService(foregroundServiceIntent)
            foregroundServiceStarted = true
            Timber.tag(TAG).d("startForegroundService: service started successfully")
        } catch (e: Exception) {
            Timber.tag(TAG).e(e, "start foreground service failed")
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
        } else {
            updateNotification()
        }
    }
    
    private fun updateNotification() {
        if (!foregroundServiceStarted || disableForegroundService) {
            Timber.tag(TAG).d("updateNotification: skipped - foregroundServiceStarted: $foregroundServiceStarted, disableForegroundService: $disableForegroundService")
            return
        }
        
        try {
            val count = activeDownloadCount.get()
            val modelName = getActiveDownloadModelName()
            
            Timber.tag(TAG).d("updateNotification: count=$count, modelName=$modelName")
            // Use the static method to update notification
            DownloadForegroundService.updateNotification(count, modelName)
        } catch (e: Exception) {
            Timber.tag(TAG).e(e, "Failed to update notification")
        }
    }
    
    private fun getActiveDownloadModelName(): String? {
        return when {
            activeDownloadModelNames.size == 1 -> activeDownloadModelNames.values.firstOrNull()
            activeDownloadModelNames.size > 1 -> null
            else -> null
        }
    }
    
    private fun addActiveDownload(modelId: String, modelName: String) {
        activeDownloadModelNames[modelId] = modelName
        updateNotification()
    }
    
    private fun removeActiveDownload(modelId: String) {
        activeDownloadModelNames.remove(modelId)
        updateNotification()
    }

    private fun setDownloadFailed(modelId: String, e: Exception) {
        Timber.tag(TAG).e(e, "[setDownloadFailed] Setting failure for $modelId. Notifying ${listeners.size} listeners.")
        val info = downloadInfoMap.getOrPut(modelId) { DownloadInfo() }
        info.downloadState = DownloadState.FAILED
        info.errorMessage = e.message
        info.errorException = e
        removeActiveDownload(modelId)
        listeners.forEach {
            Timber.tag(TAG).d("[setDownloadFailed] Notifying listener: ${it.javaClass.simpleName}")
            it.onDownloadFailed(modelId, e)
        }
    }

    /**
     * Handle repo information received from downloaders
     * @param modelId the model ID
     * @param lastModified server last modified timestamp
     * @param repoSize total repo size in bytes
     */
    private fun onRepoInfoReceived(modelId: String, lastModified: Long, repoSize: Long) {
        Timber.tag(TAG).d("[onRepoInfoReceived] modelId: $modelId, lastModified: $lastModified, repoSize: $repoSize")
        
        // Update download info with repo size if not already set
        val downloadInfo = getDownloadInfo(modelId)
        if (downloadInfo.totalSize <= 0 && repoSize > 0) {
            downloadInfo.totalSize = repoSize
            DownloadPersistentData.saveDownloadSizeTotal(ApplicationProvider.get(), modelId, repoSize)
            listeners.forEach { it.onDownloadTotalSize(modelId, repoSize) }
        }
        
        // Update upload time for update checking
        if (lastModified > 0) {
            downloadInfo.remoteUpdateTime = lastModified
            if (downloadInfo.downloadedTime > 0) {
                Timber.tag(TAG).d("[onRepoInfoReceived] downloadedTime: ${downloadInfo.downloadedTime}, lastModified: $lastModified")
            }

            if (downloadInfo.downloadedTime in 1..<lastModified) {
                // Set hasUpdate flag to true when update is available
                downloadInfo.hasUpdate = true
                listeners.forEach {
                    it.onDownloadHasUpdate(modelId, downloadInfo)
                }
            }
        }
    }


    private fun updateDownloadingProgress(
        modelId: String,
        stage: String,
        currentFile: String?,
        savedSize: Long,
        totalSize: Long
    ) {
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
        if (totalSize > 0) {
            downloadInfo.progress = savedSize.toDouble() / totalSize
            DownloadPersistentData.saveDownloadSizeSaved(ApplicationProvider.get(), modelId, savedSize)
            DownloadPersistentData.saveDownloadSizeTotal(ApplicationProvider.get(), modelId, totalSize)
            calculateDownloadSpeed(downloadInfo, savedSize)
        }
        Timber.tag(TAG).v("[updateDownloadingProgress] Notifying ${listeners.size} listeners for $modelId stage: $stage")
        listeners.forEach { it.onDownloadProgress(modelId, downloadInfo) }
        
        // Update notification with progress information
        updateNotification()
    }

    suspend fun deleteModel(item: ModelMarketItem) {
        deleteModel(item.modelId, item.currentSource, item.currentRepoPath)
    }

    suspend fun deleteModel(modelId:String, modelSource:String, modelPath:String) {
        withContext(Dispatchers.IO) {
            getDownloaderForSource(modelSource).deleteRepo(modelId)
            DownloadPersistentData.removeProgressSuspend(ApplicationProvider.get(), modelId)
            clearMmapCache(modelId)
            ModelConfig.getModelConfigDir(modelId).let {
                DownloadFileUtils.deleteDirectoryRecursively(File(it))
            }
        }
        val downloadInfo = getDownloadInfo(modelId)
        downloadInfo.downloadState = DownloadState.NOT_START
        // Reset the checked status when model is deleted
        checkedForUpdateModelIds.remove(modelId)
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

    suspend fun checkForUpdate(modelId: String?) {
        Timber.tag(TAG).v("checkForUpdate: $modelId")
        modelId?.let {
            if (!checkedForUpdateModelIds.containsKey(it)) {
                checkedForUpdateModelIds[it] = true
                getDownloaderForSource(ModelUtils.getSource(modelId)!!).checkUpdate(modelId)
            } else {
                Timber.tag(TAG).v("checkForUpdate: $modelId already checked, skipping")
            }
        }
    }

    /**
     * Reset the checked status for a modelId to allow checking for updates again.
     * This is useful for manual refresh scenarios or testing.
     */
    fun resetCheckedStatus(modelId: String) {
        checkedForUpdateModelIds.remove(modelId)
        Timber.tag(TAG).d("resetCheckedStatus: $modelId")
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
