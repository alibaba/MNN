// Created by ruoyi.sjd on 2025/1/10.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.utils

import android.annotation.SuppressLint
import android.content.Context
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.util.Log
import com.alibaba.mls.api.ApplicationProvider
import com.alibaba.mls.api.download.DownloadFileUtils
import com.alibaba.mnnllm.android.model.ModelUtils
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStream
import java.io.OutputStream
import java.nio.file.FileVisitOption
import java.nio.file.FileVisitResult
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.SimpleFileVisitor
import java.nio.file.attribute.BasicFileAttributes
import java.util.EnumSet

object FileUtils {
    const val TAG: String = "FileUtils"
    fun getAudioDuration(audioFilePath: String?): Long {
        val mmr = MediaMetadataRetriever()
        try {
            mmr.setDataSource(audioFilePath)
            val durationStr = mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
            return if (durationStr != null) durationStr.toLong() / 1000 else -1
        } catch (e: Exception) {
            Log.e(TAG, "", e)
            return 1
        } finally {
            try {
                mmr.release()
            } catch (e: IOException) {
                Log.e(TAG, "", e)
            }
        }
    }

    fun generateDestDiffusionFilePath(context: Context, sessionId: String): String {
        return generateDestFilePathKindOf(context, sessionId, "diffusion", "jpg")
    }

    fun generateDestPhotoFilePath(context: Context, sessionId: String): String {
        return generateDestFilePathKindOf(context, sessionId, "photo", "jpg")
    }

    fun generateDestAudioFilePath(context: Context, sessionId: String): String {
        return generateDestFilePathKindOf(context, sessionId, "audio", "wav")
    }

    fun generateDestRecordFilePath(context: Context, sessionId: String): String {
        return generateDestFilePathKindOf(context, sessionId, "record", "wav")
    }

    fun generateDestImageFilePath(context: Context, sessionId: String): String {
        return generateDestFilePathKindOf(context, sessionId, "image", "jpg")
    }

    private fun generateDestFilePathKindOf(
        context: Context,
        sessionId: String,
        kind: String,
        extension: String
    ): String {
        val path =
            context.filesDir.absolutePath + "/history/" + sessionId + "/" + kind + "_" + System.currentTimeMillis() + "." + extension
        ensureParentDirectoriesExist(File(path))
        return path
    }

    @JvmStatic
    fun getSessionResourceBasePath(context: Context, sessionId: String): String {
        return context.filesDir.absolutePath + "/" + sessionId
    }

    fun ensureParentDirectoriesExist(file: File) {
        val parentDir = file.parentFile
        if (parentDir != null && !parentDir.exists()) {
            parentDir.mkdirs()
        }
    }

    @Throws(IOException::class)
    fun copyFileUriToPath(context: Context, fileUri: Uri, destFilePath: String): File {
        var inputStream: InputStream? = null
        var outputStream: OutputStream? = null
        try {
            // Open an InputStream from the Uri
            inputStream = context.contentResolver.openInputStream(fileUri)
            requireNotNull(inputStream) { "Unable to open InputStream from Uri" }
            // Create the destination file
            val destinationFile = File(destFilePath)
            ensureParentDirectoriesExist(destinationFile)
            outputStream = Files.newOutputStream(destinationFile.toPath())

            // Buffer for data transfer
            val buffer = ByteArray(4096)
            var bytesRead: Int
            while ((inputStream.read(buffer).also { bytesRead = it }) != -1) {
                outputStream.write(buffer, 0, bytesRead)
            }
            outputStream.flush()

            return destinationFile
        } finally {
            try {
                inputStream?.close()
            } catch (ignored: Exception) {
            }
            try {
                outputStream?.close()
            } catch (ignored: Exception) {
            }
        }
    }

    fun saveStringToFile(context: Context, fileName: String?, data: String) {
        var fileOutputStream: FileOutputStream? = null
        try {
            fileOutputStream = context.openFileOutput(fileName, Context.MODE_PRIVATE)
            fileOutputStream.write(data.toByteArray())
            fileOutputStream.close()
            println("File saved successfully")
        } catch (e: Exception) {
            Log.e(TAG, "saveStringToFile error", e)
        } finally {
            if (fileOutputStream != null) {
                try {
                    fileOutputStream.close()
                } catch (e: Exception) {
                    e.printStackTrace()
                }
            }
        }
    }

    fun getMmapDir(modelId: String, isModelScope: Boolean): String {
        var rootCacheDir =
            ApplicationProvider.get().filesDir.toString() + "/tmps/" + ModelUtils.safeModelId(
                modelId
            )
        if (isModelScope) {
            rootCacheDir = "$rootCacheDir/modelscope"
        }
        return rootCacheDir
    }

    fun clearMmapCache(modelId: String):Boolean {
        return DownloadFileUtils.deleteDirectoryRecursively(File(getMmapDir(modelId, true))) ||
        DownloadFileUtils.deleteDirectoryRecursively(File(getMmapDir(modelId, false)))
    }

    fun getPathForUri(uri: Uri): String? {
        if ("file" == uri.scheme) {
            return uri.path
        }
        return null
    }

    @SuppressLint("DefaultLocale")
    fun formatFileSize(size: Long): String {
        val kb = 1024L
        val mb = kb * 1024L
        val gb = mb * 1024L

        return when {
            size >= gb -> String.format("%.2f GB", size.toFloat() / gb)
            size >= mb -> String.format("%.2f MB", size.toFloat() / mb)
            size >= kb -> String.format("%.2f KB", size.toFloat() / kb)
            else -> "$size B"
        }
    }

    fun getFileSizeString(file: File?): String {
        val size = getFileSize(file)
        return formatFileSize(size?:0)
    }

    fun getFileSize(file: File?): Long {
        val start = file?.toPath() ?: return 0L
        val visited = mutableSetOf<Path>()
        var total = 0L

        Files.walkFileTree(start,
            EnumSet.of(FileVisitOption.FOLLOW_LINKS),
            Integer.MAX_VALUE,
            object : SimpleFileVisitor<Path>() {
                override fun preVisitDirectory(dir: Path, attrs: BasicFileAttributes): FileVisitResult {
                    val real = dir.toRealPath()
                    return if (!visited.add(real)) {
                        FileVisitResult.SKIP_SUBTREE
                    } else {
                        FileVisitResult.CONTINUE
                    }
                }
                override fun visitFile(file: Path, attrs: BasicFileAttributes): FileVisitResult {
                    if (attrs.isRegularFile) {
                        total += attrs.size()
                    }
                    return FileVisitResult.CONTINUE
                }
            }
        )
        return total
    }
}

