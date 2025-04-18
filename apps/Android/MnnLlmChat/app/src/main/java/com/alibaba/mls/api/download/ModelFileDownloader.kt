// Created by ruoyi.sjd on 2024/12/18.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mls.api.download

import android.util.Log
import com.alibaba.mls.api.HfApiException
import com.alibaba.mls.api.download.DownloadFileUtils.createSymlink
import com.alibaba.mls.api.download.DownloadFileUtils.moveWithPermissions
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.IOException
import java.io.RandomAccessFile
import java.util.concurrent.TimeUnit

class ModelFileDownloader {
    private val client: OkHttpClient = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .followRedirects(false)
        .followSslRedirects(false) //                .addInterceptor(OkHttpUtils.createLoggingInterceptor())
        .build()

    @Throws(HfApiException::class, DownloadPausedException::class, IOException::class)
    fun downloadFile(
        fileDownloadTask: FileDownloadTask,
        fileDownloadListener: FileDownloadListener
    ) {
        // Create necessary directories
        Log.d(TAG, "downloadFile inner")
        fileDownloadTask.pointerPath!!.parentFile.mkdirs()
        fileDownloadTask.blobPath!!.parentFile.mkdirs()

        if (fileDownloadTask.pointerPath!!.exists()) {
            Log.d(TAG, "DownloadFile " + fileDownloadTask.relativePath + " already exists")
            return
        }

        if (fileDownloadTask.blobPath!!.exists()) {
            createSymlink(
                fileDownloadTask.blobPath.toString(),
                fileDownloadTask.pointerPath.toString()
            )
            Log.d(
                TAG,
                "DownloadFile " + fileDownloadTask.relativePath + " already exists just create symlink"
            )
            return
        }
        synchronized(this) {
            val hfFileMetadata = fileDownloadTask.hfFileMetadata
            downloadToTmpAndMove(
                fileDownloadTask,
                fileDownloadTask.blobPathIncomplete!!,
                fileDownloadTask.blobPath!!,
                hfFileMetadata!!.location!!,
                hfFileMetadata.size,
                fileDownloadTask.relativePath, false, fileDownloadListener
            )
            createSymlink(
                fileDownloadTask.blobPath!!.toPath(),
                fileDownloadTask.pointerPath!!.toPath()
            )
        }
    }

    @Throws(HfApiException::class, DownloadPausedException::class)
    private fun downloadToTmpAndMove(
        fileDownloadTask: FileDownloadTask,
        incompletePath: File,
        destinationPath: File,
        urlToDownload: String,
        expectedSize: Long,
        fileName: String?,
        forceDownload: Boolean,
        fileDownloadListener: FileDownloadListener
    ) {
        var urlToDownload = urlToDownload
        if (destinationPath.exists() && !forceDownload) {
            return
        }
        if (incompletePath.exists() && forceDownload) {
            incompletePath.delete()
        }

        if (fileDownloadTask.downloadedSize >= expectedSize) {
            return
        }
        val requestBuilder = Request.Builder()
            .url(urlToDownload)
            .get()
        val request: Request = requestBuilder.build()
        try {
            client.newCall(request).execute().use { response ->
                Log.d(TAG, "response code: " + response.code)
                for (header in response.headers.names()) {
                    Log.d(
                        TAG,
                        "downloadToTmpAndMove response header: " + header + ": " + response.header(
                            header
                        )
                    )
                }
                if (response.code == 302 || response.code == 303) {
                    urlToDownload = response.header("Location")!!
                }
            }
        } catch (e: IOException) {
            throw HfApiException("get header error" + e.message)
        }
        Log.d(
            TAG,
            "downloadToTmpAndMove urlToDownload: $urlToDownload to file: $incompletePath to destination: $destinationPath"
        )
        val maxRetry = 10
        if (fileDownloadTask.downloadedSize < expectedSize) {
            for (i in 0 until maxRetry) {
                try {
                    Log.d(
                        TAG,
                        "downloadChunk try the $i turn"
                    )
                    downloadChunk(
                        fileDownloadTask,
                        urlToDownload,
                        incompletePath,
                        expectedSize,
                        fileName,
                        fileDownloadListener
                    )
                    Log.d(
                        TAG,
                        "downloadChunk try the $i turn finish"
                    )
                    break
                } catch (e: DownloadPausedException) {
                    throw e
                } catch (e: Exception) {
                    if (i == maxRetry - 1) {
                        throw e
                    } else {
                        Log.e(TAG, "downloadChunk failed sleep and retrying: " + e.message)
                        try {
                            Thread.sleep(1000)
                        } catch (ex: InterruptedException) {
                            throw RuntimeException(ex)
                        }
                    }
                }
            }
        }
        moveWithPermissions(incompletePath, destinationPath)
    }

    @Throws(HfApiException::class, DownloadPausedException::class)
    private fun downloadChunk(
        fileDownloadTask: FileDownloadTask,
        url: String,
        tempFile: File,
        expectedSize: Long,
        displayedFilename: String?,
        fileDownloadListener: FileDownloadListener?
    ) {
        val requestBuilder = Request.Builder()
            .url(url)
            .get()
            .header("Accept-Encoding", "identity")
        if (fileDownloadTask.downloadedSize >= expectedSize) {
            return
        }
        var downloadedBytes = fileDownloadTask.downloadedSize

        if (fileDownloadTask.downloadedSize > 0) {
            requestBuilder.header("Range", "bytes=" + fileDownloadTask.downloadedSize + "-")
        }
        Log.d(
            TAG,
            "resume size: " + fileDownloadTask.downloadedSize + " expectedSize: " + expectedSize
        )
        val request: Request = requestBuilder.build()
        try {
            client.newCall(request).execute().use { response ->
                Log.d(
                    TAG,
                    "downloadChunk response: success: " + response.isSuccessful + " code: " + response.code
                )
                if (response.isSuccessful || response.code == 416) {
                    response.body!!
                        .byteStream().use { `is` ->
                            RandomAccessFile(tempFile, "rw").use { raf ->
                                raf.seek(fileDownloadTask.downloadedSize)
                                val buffer = ByteArray(8192)
                                var bytesRead: Int
                                while ((`is`.read(buffer).also { bytesRead = it }) != -1) {
                                    raf.write(buffer, 0, bytesRead)
                                    downloadedBytes += bytesRead.toLong()
                                    fileDownloadTask.downloadedSize += bytesRead.toLong()
                                    if (fileDownloadListener != null) {
                                        val paused = fileDownloadListener.onDownloadDelta(
                                            displayedFilename,
                                            downloadedBytes,
                                            expectedSize,
                                            bytesRead.toLong()
                                        )
                                        if (paused) {
                                            throw DownloadPausedException("Download paused")
                                        }
                                    }
                                }
                            }
                        }
                } else {
                    Log.e(TAG, "downloadChunk error HfApiException " + response.code)
                    throw HfApiException("HTTP error: ${response.code}")
                }
            }
        } catch (e: IOException) {
            Log.e(TAG, "downloadChunk error IOException", e)
            throw HfApiException("Connection error: " + e.message)
        }
    }

    interface FileDownloadListener {
        fun onDownloadDelta(
            fileName: String?,
            downloadedBytes: Long,
            totalBytes: Long,
            delta: Long
        ): Boolean
    }

    companion object {
        const val TAG: String = "RemoteModelDownloader"
    }
}
