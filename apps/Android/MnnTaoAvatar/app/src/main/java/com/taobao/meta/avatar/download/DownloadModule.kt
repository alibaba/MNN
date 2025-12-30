// Created by ruoyi.sjd on 2025/4/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.taobao.meta.avatar.download

import android.app.Activity
import android.util.Log
import com.alibaba.mls.api.download.DownloadInfo
import com.alibaba.mls.api.download.DownloadListener
import com.alibaba.mls.api.download.RepoDownloadSate
import com.alibaba.mls.api.download.ModelDownloadManager
import com.alibaba.mnnllm.android.utils.FileUtils.formatFileSize
import com.taobao.meta.avatar.debug.DebugModule.Companion.DEBUG_USE_PRIVATE
import java.util.Collections
import java.util.concurrent.ConcurrentHashMap

/**
 * responsible for downloading avatar resources from modelscope,huggingface, etc.
 */
class DownloadModule(private val context: Activity) {

    //define the minimal repos to download
    private val reposToDownload = listOf(
        "ModelScope/MNN/Qwen2.5-1.5B-Instruct-MNN",
        "ModelScope/MNN/UniTalker-MNN",
        "ModelScope/MNN/bert-vits2-MNN",
        "ModelScope/MNN/TaoAvatar-NNR-MNN",
        "ModelScope/MNN/sherpa-mnn-streaming-zipformer-bilingual-zh-en-2023-02-20",
        "ModelScope/MNN/sherpa-mnn-streaming-zipformer-en-2023-02-21",
        "ModelScope/MNN/supertonic-tts-mnn"
    )

    private val finishedSet = Collections.synchronizedSet(mutableSetOf<String>())
    private val sizeMap = ConcurrentHashMap<String, Long>()
    private val savedMap = ConcurrentHashMap<String, Long>()
    private var downloadCallback: DownloadCallback? = null
    private var lastProgressTime: Long = 0
    private var downloadSpeedStr = "0Bps"
    private val modelsDownloadManager = ModelDownloadManager(context)
    private var lastProgressBytes: Long = 0
    private val downloadListener = object : DownloadListener {
        override fun onDownloadStart(modelId: String) {
        }

        override fun onDownloadFailed(modelId: String, e: Exception) {
        }

        override fun onDownloadProgress(modelId: String, progress: DownloadInfo) {
//            Log.d(TAG, "onDownloadProgress: $modelId $progress isDownloading: ${progress.isDownloading()}")
            if (!progress.isDownloading()) {
                return
            }
            if (!sizeMap.containsKey(modelId)) {
                sizeMap[modelId] = progress.totalSize
            }
            savedMap[modelId] = progress.savedSize
//            Log.d(TAG, "onDownloadProgress sizeMap: ${sizeMap} size: ${sizeMap.size} savedMap: ${savedMap}")
            if (sizeMap.size == reposToDownload.size) {
                val currentTime = System.currentTimeMillis()
                val timeElapsed = currentTime - lastProgressTime
                val totalLength = sizeMap.values.sum()
                val totalProgress = savedMap.values.sum()
                val bytesDownloaded = totalProgress - lastProgressBytes
                if (timeElapsed > 1000 && bytesDownloaded > 0) {
                    val downloadSpeed = bytesDownloaded / timeElapsed * 1000 // bytes per second
                    downloadSpeedStr = "${formatFileSize(downloadSpeed)}ps"
                    lastProgressTime = currentTime
                    lastProgressBytes = totalProgress
                }
//                Log.d(TAG, "onDownloadProgress progress: ${totalProgress.toDouble() / totalLength} totalProgress: $totalProgress totalLength: $totalLength downloadSpeedStr: $downloadSpeedStr")
                downloadCallback?.onDownloadProgress(totalProgress.toDouble() / totalLength,
                                                      totalProgress,
                                                      totalLength,
                                                      downloadSpeedStr
                    )
            }
        }

        override fun onDownloadFinished(modelId: String, path: String) {
            finishedSet.add(modelId)
            Log.d(TAG, "onDownloadFinished: $modelId path: $path finishedSet:$finishedSet reposToDownload: ${reposToDownload}" )
            if (finishedSet.size == reposToDownload.size) {
                downloadCallback?.onDownloadComplete(true, null)
            }
        }

        override fun onDownloadPaused(modelId: String) {
        }

        override fun onDownloadFileRemoved(modelId: String) {
        }
    }

    init {
        modelsDownloadManager.setListener(downloadListener)
    }

    fun isDownloadComplete(): Boolean {
        val downloadComplete =  reposToDownload.all {
            modelsDownloadManager.getDownloadInfo(it).downlodaState == RepoDownloadSate.COMPLETED
        }
        Log.d(TAG, "downloadComplete: $downloadComplete")
        return downloadComplete
    }

    fun download() {
        Log.d(TAG , "start download")
        reposToDownload.forEach {
            val downloadInfo = modelsDownloadManager.getDownloadInfo(it)
            if (downloadInfo.canDownload()) {
                modelsDownloadManager.startDownload(it)
            } else if (downloadInfo.isComplete()) {
                Log.d(TAG , "already downloaded: $it size: ${formatFileSize( downloadInfo.totalSize)}")
                sizeMap[it] = downloadInfo.totalSize
                savedMap[it] = downloadInfo.totalSize
                finishedSet.add(it)
            }
        }
        // Check immediate completion if all were already downloaded
        if (finishedSet.size == reposToDownload.size) {
             downloadCallback?.onDownloadComplete(true, null)
        }
    }

    fun getDownloadPath(): String {
        if (DEBUG_USE_PRIVATE) {
            return context.filesDir.absolutePath + "/metahuman"
        } else {
            return context.filesDir.absolutePath + "/.mnnmodels/modelscope"
        }
    }

    fun setDownloadCallback(callback: DownloadCallback) {
        downloadCallback = callback
    }

    companion object {
        const val TAG = "DownloadModule"
    }

}
