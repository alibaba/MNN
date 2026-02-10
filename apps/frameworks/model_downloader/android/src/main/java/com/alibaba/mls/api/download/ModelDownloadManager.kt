// Created by ruoyi.sjd on 2024/12/24.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mls.api.download

import android.content.Context
import com.alibaba.mls.api.ApplicationProvider
import com.alibaba.mls.api.download.hf.HfModelDownloader
import com.alibaba.mls.api.download.ms.MsModelDownloader
import com.alibaba.mls.api.download.DownloadCoroutineManager
import com.alibaba.mls.api.download.DownloadPersistentData
import java.io.File

/**
 * Minimal ModelDownloadManager for framework
 * Apps can extend or wrap this class for additional functionality
 */
class ModelDownloadManager(context: Context) : ModelRepoDownloader.ModelRepoDownloadCallback {

    private val cacheDir: String = context.filesDir.absolutePath + "/.mnnmodels"
    private val hfDownloader: HfModelDownloader
    private val msDownloader: MsModelDownloader

    private var downloadListener: DownloadListener? = null
    private val downloadListeners: MutableList<DownloadListener> = java.util.concurrent.CopyOnWriteArrayList()

    init {
        hfDownloader = HfModelDownloader(this, cacheDir)
        msDownloader = MsModelDownloader(this, cacheDir)
    }
    
    // Cache for download info to support progress and pause states
    private val downloadInfoMap = java.util.concurrent.ConcurrentHashMap<String, DownloadInfo>()
    
    // Helper to get or create info
    private fun getOrCreateInfo(modelId: String): DownloadInfo {
        return downloadInfoMap.getOrPut(modelId) { DownloadInfo() }
    }

    fun setListener(listener: DownloadListener?) {
        if (this.downloadListener != null) {
            removeListener(this.downloadListener!!)
        }
        if (listener != null) {
            addListener(listener)
        }
        this.downloadListener = listener
    }

    fun addListener(listener: DownloadListener) {
        if (!downloadListeners.contains(listener)) {
            downloadListeners.add(listener)
        }
    }

    fun removeListener(listener: DownloadListener) {
        downloadListeners.remove(listener)
    }

    fun setDownloadListener(listener: DownloadListener?) {
        this.downloadListener = listener
    }

    fun startDownload(modelId: String) {
        // Immediately trigger preparing state to update UI
        val info = getOrCreateInfo(modelId)
        info.downloadState = DownloadState.PREPARING
        info.downlodaState = DownloadState.PREPARING
        info.progress = 0.0
        downloadListeners.forEach { it.onDownloadStart(modelId) }
        
        when {
            modelId.startsWith("HuggingFace/") || modelId.startsWith("Huggingface/") -> {
                hfDownloader.download(modelId)
            }
            modelId.startsWith("ModelScope/") -> {
                msDownloader.download(modelId)
            }
            else -> {
                // Default to ModelScope for unprefixed models
                // This supports models like "taobao-mnn/..." from ModelScope
                msDownloader.download(modelId)
            }
        }
    }

    fun isDownloading(modelId: String): Boolean {
        // Simplified - apps can override
        return false
    }

    fun pauseDownload(modelId: String) {
        hfDownloader.pause(modelId)
        msDownloader.pause(modelId)
    }

    fun pauseAllDownloads() {
        // Simplified - apps can track active downloads and pause them
    }

    fun deleteModel(modelId: String) {
        val file = getDownloadedFile(modelId)
        file?.deleteRecursively()
        downloadListeners.forEach { it.onDownloadFileRemoved(modelId) }
    }

    fun downloadQnnLibs(): Boolean {
        // Stub - apps can implement QNN library download if needed
        return false
    }

    fun tryStartForegroundService() {
        // Stub - apps can implement foreground service management if needed
    }

    fun checkForUpdate(modelId: String?) {
        if (modelId == null) return
        DownloadCoroutineManager.launchDownload {
             if (modelId.startsWith("HuggingFace/") || modelId.startsWith("Huggingface/")) {
                 hfDownloader.checkUpdate(modelId)
             } else {
                 // Default to ModelScope for unprefixed models
                 msDownloader.checkUpdate(modelId)
             }
        }
    }

    fun getDownloadInfo(modelId: String): DownloadInfo {
        // First check if we have active/cached info
        val cachedInfo = downloadInfoMap[modelId]
        if (cachedInfo != null) {
             // Always return cached info if it exists, regardless of state
             // This ensures modifications (like totalSize from ModelMarketViewModel) are preserved
             return cachedInfo
        }

        val isDownloaded = if (modelId.startsWith("HuggingFace/") || modelId.startsWith("Huggingface/")) {
            hfDownloader.isDownloaded(modelId)
        } else {
            // Default to ModelScope for unprefixed models
            msDownloader.isDownloaded(modelId)
        }
        
        val info = if (isDownloaded) {
            val totalSize = DownloadPersistentData.getDownloadSizeTotal(ApplicationProvider.get(), modelId)
            DownloadInfo(downloadState = DownloadState.DOWNLOAD_SUCCESS, downlodaState = DownloadState.DOWNLOAD_SUCCESS, progress = 1.0, totalSize = totalSize)
        } else {
            val totalSize = DownloadPersistentData.getDownloadSizeTotal(ApplicationProvider.get(), modelId)
            if (totalSize > 0) {
                 val savedSize = getRealDownloadSize(modelId)
                 val progress = if (totalSize > 0) savedSize.toDouble() / totalSize else 0.0
                 val state = if (savedSize > 0) DownloadState.PAUSED else DownloadState.NOT_START
                 DownloadInfo(downloadState = state, downlodaState = state, progress = progress, totalSize = totalSize, savedSize = savedSize)
            } else {
                 DownloadInfo(downloadState = DownloadState.NOT_START, downlodaState = DownloadState.NOT_START)
            }
        }
        // Cache the info so modifications are preserved
        downloadInfoMap[modelId] = info
        return info
    }

    fun getDownloadState(modelId: String): RepoDownloadSate {
        val info = getDownloadInfo(modelId)
        return RepoDownloadSate(
            modelId = modelId,
            state = info.downloadState,
            downlodaState = info.downloadState,
            progress = info.progress,
            totalSize = info.totalSize,
            savedSize = info.savedSize,
            errorMessage = info.errorMessage
        )
    }

    fun getDownloadedFile(modelId: String): File? {
        val hfPath = hfDownloader.getDownloadPath(modelId)
        if (hfPath.exists()) return hfPath

        val msPath = msDownloader.getDownloadPath(modelId)
        if (msPath.exists()) return msPath

        return null
    }

    // ModelRepoDownloadCallback implementation
    override fun onDownloadFailed(modelId: String, e: Exception) {
        val info = getOrCreateInfo(modelId)
        info.downloadState = DownloadState.FAILED
        info.downlodaState = DownloadState.FAILED
        info.errorException = e
        info.errorMessage = e.message
        downloadListeners.forEach { it.onDownloadFailed(modelId, e) }
    }

    override fun onDownloadTaskAdded() {
        // No direct mapping
    }

    override fun onDownloadTaskRemoved() {
        // No direct mapping
    }

    override fun onDownloadPaused(modelId: String) {
        val info = getOrCreateInfo(modelId)
        info.downloadState = DownloadState.PAUSED
        info.downlodaState = DownloadState.PAUSED
        downloadListeners.forEach { it.onDownloadPaused(modelId) }
    }

    override fun onDownloadFileFinished(modelId: String, absolutePath: String) {
        val info = getOrCreateInfo(modelId)
        info.downloadState = DownloadState.DOWNLOAD_SUCCESS
        info.downlodaState = DownloadState.DOWNLOAD_SUCCESS
        info.progress = 1.0
        info.hasUpdate = false
        
        // Save total size to persistent data for future retrievals
        if (info.totalSize > 0) {
            DownloadPersistentData.saveDownloadSizeTotal(ApplicationProvider.get(), modelId, info.totalSize)
        }

        downloadListeners.forEach { it.onDownloadFinished(modelId, absolutePath) }
    }

    // Speed tracking helper
    private data class SpeedTracker(var lastTime: Long = 0, var lastSaved: Long = 0)
    private val speedTrackers = java.util.concurrent.ConcurrentHashMap<String, SpeedTracker>()

    override fun onDownloadingProgress(
        modelId: String,
        stage: String,
        currentFile: String?,
        saved: Long,
        total: Long
    ) {
        val progress = if (total > 0) saved.toDouble() / total else 0.0
        
        // Calculate speed with throttling (every 1 second)
        val currentTime = System.currentTimeMillis()
        val info = getOrCreateInfo(modelId)
        val tracker = speedTrackers.getOrPut(modelId) { SpeedTracker(currentTime, saved) }
        
        if (currentTime - tracker.lastTime >= 1000) {
           val timeDelta = currentTime - tracker.lastTime
           val bytesDelta = saved - tracker.lastSaved
           if (timeDelta > 0 && bytesDelta >= 0) {
               val speed = bytesDelta * 1000 / timeDelta
               val speedStr = if (speed < 1024) "$speed B" else if (speed < 1024 * 1024) String.format("%.2f KB", speed / 1024.0) else String.format("%.2f MB", speed / (1024.0 * 1024.0))
               info.speedInfo = "$speedStr/s"
           }
           // Update tracker
           tracker.lastTime = currentTime
           tracker.lastSaved = saved
        }
        
        info.downloadState = DownloadState.DOWNLOADING
        info.downlodaState = DownloadState.DOWNLOADING
        info.progress = progress
        info.savedSize = saved
        // Only update totalSize if it's valid (prevent flickering if total is 0 during preparing)
        if (total > 0) {
            info.totalSize = total
        }
        info.currentFile = currentFile
        info.progressStage = stage
        info.lastProgressUpdateTime = currentTime
        
        downloadListeners.forEach { it.onDownloadProgress(modelId, info) }
        
        // Also trigger onDownloadStart if it's the first progress update (simplified)
        if (saved == 0L) {
             downloadListeners.forEach { it.onDownloadStart(modelId) }
        }
    }

    override fun onRepoInfo(modelId: String, lastModified: Long, repoSize: Long) {
        val info = getOrCreateInfo(modelId)
        info.totalSize = repoSize
        info.remoteUpdateTime = lastModified
        
        if (info.downloadState == DownloadState.DOWNLOAD_SUCCESS) {
            info.hasUpdate = true
            downloadListeners.forEach { it.onDownloadHasUpdate(modelId, info) }
        }
        
        downloadListeners.forEach { it.onDownloadTotalSize(modelId, repoSize) }
    }

    private fun getDownloader(modelId: String): ModelRepoDownloader {
        return if (modelId.startsWith("HuggingFace/") || modelId.startsWith("Huggingface/")) {
            hfDownloader
        } else {
             msDownloader
        }
    }

    fun getRealDownloadSize(modelId: String): Long {
         val savedSize = DownloadPersistentData.getDownloadSizeSaved(ApplicationProvider.get(), modelId)
         if (savedSize > 0) return savedSize
         
         val downloader = getDownloader(modelId)
         val realFile = downloader.repoModelRealFile(modelId)
         if (realFile.exists()) {
              return FileUtils.getFileSize(realFile)
         }
         return 0L
    }

    companion object {
        const val REQUEST_CODE_POST_NOTIFICATIONS = 1001  // Stub constant for foreground service notifications

        @Volatile
        private var instance: ModelDownloadManager? = null

        @JvmStatic
        fun getInstance(context: Context): ModelDownloadManager {
            return instance ?: synchronized(this) {
                instance ?: ModelDownloadManager(context.applicationContext).also {
                    instance = it
                }
            }
        }

        @JvmStatic
        fun getInstance(): ModelDownloadManager {
            return getInstance(ApplicationProvider.get())
        }
    }
}
