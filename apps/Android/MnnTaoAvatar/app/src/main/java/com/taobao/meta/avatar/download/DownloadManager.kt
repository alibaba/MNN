// Created by ruoyi.sjd on 2025/3/31.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.taobao.meta.avatar.download

import android.content.Context
import android.util.Log
import com.alibaba.mnnllm.android.utils.FileUtils.formatFileSize
import com.liulishuo.okdownload.DownloadListener
import com.liulishuo.okdownload.DownloadTask
import com.liulishuo.okdownload.OkDownload
import com.liulishuo.okdownload.core.breakpoint.BreakpointInfo
import com.liulishuo.okdownload.core.breakpoint.BreakpointStoreOnSQLite
import com.liulishuo.okdownload.core.cause.EndCause
import com.liulishuo.okdownload.core.cause.ResumeFailedCause
import com.liulishuo.okdownload.core.dispatcher.DownloadDispatcher
import java.io.File

class DownloadManager(private val context: Context) {

    companion object {
        const val TAG = "DownloadManager"
        private const val MODEL_URL = "https://meta.alicdn.com/data/mnn/avatar/qwen2.5-1.5b-instruct-int8-private.zip"
    }

    init {
        val builder = OkDownload.Builder(context)
            .downloadStore(BreakpointStoreOnSQLite(context))
        OkDownload.setSingletonInstance(builder.build())
        DownloadDispatcher.setMaxParallelRunningCount(3);
    }

    private var downloadCallback: DownloadCallback? = null
    private var lastProgressTime: Long = 0
    private var lastProgressBytes: Long = 0

    fun getDownloadPath(): String {
        return context.filesDir.absolutePath + "/metahuman"
    }

    fun getDownloadSuccessFlagPath(): String {
        return context.filesDir.absolutePath + "/metahuman/success"
    }

    fun setDownloadCallback(callback: DownloadCallback) {
        downloadCallback = callback
    }

    fun isDownloadComplete():Boolean {
        val file = File(getDownloadSuccessFlagPath())
        return file.exists()
    }

    fun download() {
        val targetFile = File(getDownloadPath() + "_tmp")
        val url = MODEL_URL
        val filename = "metahuman-model.zip"
        val task = DownloadTask.Builder(url, targetFile)
            .setFilename(filename)
            .setConnectionCount(1)
            .setMinIntervalMillisCallbackProcess(100)
            .setPassIfAlreadyCompleted(true)
            .build()
        var downloadSpeedStr = "0Bps"
        task.enqueue(object : DownloadListener {
            override fun taskStart(task: DownloadTask) {
                Log.d(TAG, "taskStart")
            }

            override fun connectTrialStart(
                task: DownloadTask,
                requestHeaderFields: MutableMap<String, MutableList<String>>
            ) {
                Log.d(TAG, "connectTrialStart")
                downloadCallback?.onDownloadStart()
            }

            override fun connectTrialEnd(
                task: DownloadTask,
                responseCode: Int,
                responseHeaderFields: MutableMap<String, MutableList<String>>
            ) {
                Log.d(TAG, "connectTrialEnd")
            }

            override fun downloadFromBeginning(
                task: DownloadTask,
                info: BreakpointInfo,
                cause: ResumeFailedCause
            ) {
                Log.d(TAG, "downloadFromBeginning cause: $cause totalFileLength: ${task.info?.totalLength} " )
            }

            override fun downloadFromBreakpoint(task: DownloadTask, info: BreakpointInfo) {
                Log.d(TAG, "downloadFromBreakpoint:")
                downloadCallback?.onDownloadStart()
            }

            override fun connectStart(
                task: DownloadTask,
                blockIndex: Int,
                requestHeaderFields: MutableMap<String, MutableList<String>>
            ) {
                Log.d(TAG, "connectStart" )
            }

            override fun connectEnd(
                task: DownloadTask,
                blockIndex: Int,
                responseCode: Int,
                responseHeaderFields: MutableMap<String, MutableList<String>>
            ) {
                Log.d(TAG, "connectEnd" )
            }

            override fun fetchStart(task: DownloadTask, blockIndex: Int, contentLength: Long) {
                Log.d(TAG, "fetchStart")
            }

            override fun fetchProgress(task: DownloadTask, blockIndex: Int, increaseBytes: Long) {
                if (task.info != null) {
//                    Log.d(TAG, "Info totalLength: ${task.info?.totalLength}" +
//                            " totalOffset: ${task.info?.totalOffset} " +
//                            "blockCount  ${task.info?.blockCount} " +
//                            "realPercent: ${(task.info?.totalOffset?:0).toDouble().div((task.info?.totalLength?:1).toDouble()).times(100)}%"
//                    )
                    val progressPercent = if (task.info!!.totalLength > 0)
                        (task.info!!.totalOffset.toDouble() / task.info!!.totalLength) * 100 else 0.0

                    val currentTime = System.currentTimeMillis()
                    if (lastProgressTime == 0L) {
                        lastProgressTime = currentTime
                        lastProgressBytes = task.info!!.totalOffset
                    }
                    val timeElapsed = currentTime - lastProgressTime
                    val bytesDownloaded = task.info!!.totalOffset - lastProgressBytes
                    if (timeElapsed > 1000 && bytesDownloaded > 0) {
                        val downloadSpeed = bytesDownloaded / timeElapsed * 1000 // bytes per second
                        downloadSpeedStr = "${formatFileSize(downloadSpeed)}ps"
                        lastProgressTime = currentTime
                        lastProgressBytes = task.info!!.totalOffset
                    }
                    downloadCallback?.onDownloadProgress(progressPercent, task.info!!.totalOffset, task.info!!.totalLength, downloadSpeedStr)
                }
            }

            override fun fetchEnd(task: DownloadTask, blockIndex: Int, contentLength: Long) {
                Log.d(TAG, "fetchEnd" )
            }

            override fun taskEnd(
                task: DownloadTask,
                cause: EndCause,
                realCause: java.lang.Exception?
            ) {
                if (cause == EndCause.COMPLETED) {
                    Log.d(TAG, "download complete: 100% ")
                    downloadCallback?.onDownloadComplete(true, task.file)
                } else if (realCause != null) {
                    Log.e(TAG, "download end: $cause", realCause)
                    downloadCallback?.onDownloadError(realCause)
                } else {
                    Log.d(TAG, "download end: $cause")
                    downloadCallback?.onDownloadComplete(false,  task.file)
                }
            }
        })
    }
    fun cancelDownload() {
        Log.d(TAG, "cancelDownload")
        OkDownload.with().downloadDispatcher().cancelAll();
    }

}
