// Created by ruoyi.sjd on 2025/1/10.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.utils

import android.annotation.SuppressLint
import android.content.Context
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.util.Log
import com.alibaba.mls.api.ApplicationProvider
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStream
import java.io.OutputStream
import java.nio.file.Files
import java.util.zip.ZipFile
import android.content.res.AssetManager
import android.graphics.Bitmap
import com.alibaba.mls.api.download.DownloadFileUtils
import java.io.File

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
            context.filesDir.absolutePath + "/" + sessionId + "/" + kind + "_" + System.currentTimeMillis() + "." + extension
        ensureParentDirectoriesExist(File(path))
        return path
    }

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

    fun saveStringToFile(context: Context, fileName: String?, data: String) {
        var fileOutputStream: FileOutputStream? = null
        try {
            fileOutputStream = FileOutputStream(fileName!!)
            fileOutputStream.write(data.toByteArray())
            fileOutputStream.close()
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

    suspend fun unzipFile(zipPath: String, targetDir: String) {
        withContext(Dispatchers.IO) {
            ZipFile(zipPath).use { zipFile ->
                zipFile.entries().asSequence().forEach { entry ->
                    val entryPath = "$targetDir/${entry.name}"
                    if (entry.isDirectory) {
                        File(entryPath).mkdirs()
                    } else {
                        Log.d(TAG, "unzipFile entryPath: $entryPath")
                        File(entryPath).parentFile?.mkdirs()
                        zipFile.getInputStream(entry).use { input ->
                            File(entryPath).outputStream().use { input.copyTo(it) }
                        }
                    }
                }
            }
        }
    }

    @Throws(IOException::class)
    suspend fun copyAssetsToFilesDir(context: Context, assetsPath: String, destPath: String) {
        withContext(Dispatchers.IO) {
            val assetManager: AssetManager = context.assets
            val files: Array<String> = assetManager.list(assetsPath) ?: return@withContext

            val destDir = File(context.filesDir, destPath)
            if (!destDir.exists()) {
                destDir.mkdirs()
            }

            for (fileName in files) {
                val sourcePath = if (assetsPath.isEmpty()) fileName else "$assetsPath/$fileName"
                val destFilePath = File(destDir, fileName).absolutePath
                if (File(destFilePath).exists()) {
                    continue
                }
                if (assetManager.list(sourcePath)?.isEmpty() == true) {
                    copyAssetFile(context, sourcePath, destFilePath)
                } else {
                    copyAssetsToFilesDir(context, sourcePath, "$destPath/$fileName")
                }
            }
        }
    }

    @Throws(IOException::class)
    private fun copyAssetFile(context: Context, assetPath: String, destPath: String) {
        val inputStream: InputStream = context.assets.open(assetPath)
        val outputStream: OutputStream = FileOutputStream(destPath)

        try {
            val buffer = ByteArray(1024)
            var read: Int
            while (inputStream.read(buffer).also { read = it } != -1) {
                outputStream.write(buffer, 0, read)
            }
        } finally {
            inputStream.close()
            outputStream.close()
        }
    }

    @JvmStatic
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

    fun saveBitmapToFile(bitmap: Bitmap, filename: String) {
        val file = File(filename)
        ensureParentDirectoriesExist(file)
        try {
            FileOutputStream(file).use { fos ->
                bitmap.compress(Bitmap.CompressFormat.PNG, 90, fos)
            }
        } catch (e: IOException) {
            Log.e("CaptureBitmap", "Failed to save bitmap $filename", e)
        }
    }

    fun calculateSize(file: File): Long {
        if (!file.exists()) return 0

        if (file.isFile) return file.length()

        var size: Long = 0
        val files = file.listFiles()
        if (files != null) {
            for (childFile in files) {
                size += calculateSize(childFile)
            }
        }
        return size
    }

    fun calculateSizeString(file: File): String {
        val size = calculateSize(file)
        return formatFileSize(size)
    }

    fun clearMmapCache(modelId: String) {
        DownloadFileUtils.deleteDirectoryRecursively(File(getMmapDir(modelId, true)))
        DownloadFileUtils.deleteDirectoryRecursively(File(getMmapDir(modelId, false)))
    }

}

