package com.alibaba.mnnllm.api.openai.network.processors

import android.content.Context
import android.graphics.BitmapFactory
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import timber.log.Timber
import java.io.File
import java.io.FileOutputStream
import java.net.URL
import java.security.MessageDigest
import java.security.NoSuchAlgorithmException
import java.util.Base64

/**
 * MNN图像处理器，负责处理图像URL（Base64或网络URL），
 * 将其保存到本地缓存并返回文件路径。
 */
class MnnImageProcessor(private val context: Context) {
    private val hashToPathMap: MutableMap<String, String> = HashMap()
    private val cacheDir: String by lazy {
        val dir = File(context.externalCacheDir, "mnn_image_cache")
        if (!dir.exists()) {
            dir.mkdirs()
        }
        dir.absolutePath
    }

    init {
        ensureCacheDirExists()
    }

    private fun ensureCacheDirExists() {
        val dir = File(cacheDir)
        if (!dir.exists()) {
            if (!dir.mkdirs()) {
                logError("Failed to create cache directory: $cacheDir")
            }
        }
    }

    /**
     * 处理图像URL，可能是Base64数据URI或网络URL。
     * @param imageUrl 图像的URL或Base64数据URI
     * @return 本地文件路径，如果处理失败则返回null
     */
    suspend fun processImageUrl(imageUrl: String): String? {
        return if (imageUrl.startsWith("data:image")) {
            // 假设是Base64数据URI, e.g., data:image/jpeg;base64,/9j/4AAQSkZJRgABAQA...
            val parts = imageUrl.split(",")
            if (parts.size == 2) {
                processBase64Image(parts[1])
            } else {
                logError("Invalid Base64 data URI format: $imageUrl")
                null
            }
        } else if (imageUrl.startsWith("http://") || imageUrl.startsWith("https://")) {
            // 网络URL
            logDebug("Processing network image URL: $imageUrl")
            // TODO: 实现网络图片下载和缓存逻辑
            // 临时返回原始URL或一个占位符，或者下载它
            downloadAndProcessNetworkImage(imageUrl)
        } else if (File(imageUrl).exists()) {
            // 已经是本地文件路径
            logDebug("Image is already a local file: $imageUrl")
            imageUrl
        } else {
            logError("Unsupported image URL format: $imageUrl")
            null
        }
    }

    private suspend fun downloadAndProcessNetworkImage(urlString: String): String? =
        withContext(Dispatchers.IO) {
            try {
                logDebug("Downloading image from: $urlString")
                val url = URL(urlString)
                val connection = url.openConnection()
                connection.connect()
                val inputStream = connection.getInputStream()
                val imageData = inputStream.readBytes()
                inputStream.close()

                if (imageData.isEmpty()) {
                    logError("Downloaded image data is empty from: $urlString")
                    return@withContext null
                }
                logDebug("Successfully downloaded image data, byte length: ${imageData.size}")

                // 使用URL的哈希或文件名作为缓存键
                val hash = calculateSHA256(urlString) // Hash the URL itself for caching
                logDebug("Calculated hash for URL '$urlString': $hash")

                if (hashToPathMap.containsKey(hash)) {
                    val cachedPath = hashToPathMap[hash]
                    if (cachedPath != null && File(cachedPath).exists()) {
                        logDebug("Found cached image for URL '$urlString': $cachedPath")
                        return@withContext cachedPath
                    }
                }

                val fileName =
                    "img_" + hash.substring(0, 8) + "." + getFileExtensionFromUrl(urlString)
                val filePath = File(cacheDir, fileName).absolutePath
                logDebug("Creating image file from URL: $filePath")

                saveImageDataToFile(imageData, filePath, hash)

            } catch (e: Exception) {
                logError(
                    "Error downloading or processing network image '$urlString': ${e.message}",
                    e
                )
                null
            }
        }

    private fun getFileExtensionFromUrl(urlString: String): String {
        return try {
            val path = URL(urlString).path
            if (path.contains(".")) path.substringAfterLast('.', "jpg") else "jpg"
        } catch (e: Exception) {
            "jpg" // Default extension
        }
    }

    private suspend fun processBase64Image(base64Data: String): String? =
        withContext(Dispatchers.IO) {
            try {
                logDebug("Processing base64 image data, length: ${base64Data.length}")
                if (base64Data.isEmpty()) {
                    logError("Invalid base64 data: empty")
                    return@withContext null
                }

                val hash = calculateSHA256(base64Data)
                logDebug("Calculated base64 image hash: $hash")

                if (hashToPathMap.containsKey(hash)) {
                    val cachedPath = hashToPathMap[hash]
                    if (cachedPath != null && File(cachedPath).exists()) {
                        logDebug("Found cached base64 image: $cachedPath")
                        return@withContext cachedPath
                    }
                }

                val imageData = Base64.getDecoder().decode(base64Data)
                logDebug("Successfully decoded base64 data, byte length: ${imageData.size}")

                val fileName = "img_" + hash.substring(0, 8) + ".jpg" // Assume jpg for base64
                val filePath = File(cacheDir, fileName).absolutePath
                logDebug("Creating image file from base64: $filePath")

                saveImageDataToFile(imageData, filePath, hash)

            } catch (e: IllegalArgumentException) {
                logError("Failed to decode base64 data: ${e.message}", e)
                null
            } catch (e: Exception) {
                logError("Error processing base64 image: ${e.message}", e)
                null
            }
        }

    private suspend fun saveImageDataToFile(
        imageData: ByteArray,
        filePath: String,
        hash: String
    ): String? = withContext(Dispatchers.IO) {
        val imageFile = File(filePath)
        try {
            FileOutputStream(imageFile).use { fos ->
                fos.write(imageData)
            }
            logDebug("Successfully saved image file: $filePath")

            if (isValidImageFile(filePath)) {
                hashToPathMap[hash] = filePath
                logDebug("Saved new image to cache: $filePath (hash: $hash)")
                filePath
            } else {
                logError("Invalid image data at $filePath, deleting.")
                imageFile.delete()
                null
            }
        } catch (e: Exception) {
            logError("Failed to save image file '$filePath': ${e.message}", e)
            imageFile.delete() // Attempt to clean up
            null
        }
    }

    private fun isValidImageFile(filePath: String): Boolean {
        return try {
            val options = BitmapFactory.Options()
            options.inJustDecodeBounds = true
            BitmapFactory.decodeFile(filePath, options)
            options.outWidth > 0 && options.outHeight > 0
        } catch (e: Exception) {
            logError("Error validating image '$filePath': ${e.message}", e)
            false
        }
    }

    private fun calculateSHA256(input: String): String {
        return try {
            val digest = MessageDigest.getInstance("SHA-256")
            val hashBytes = digest.digest(input.toByteArray(Charsets.UTF_8))
            hashBytes.joinToString("") { "%02x".format(it) }
        } catch (e: NoSuchAlgorithmException) {
            logError("SHA-256 algorithm not found", e)
            // Fallback or rethrow, for now, use a simple hash of the input string for uniqueness
            input.hashCode().toString()
        }
    }

    fun cleanupCache(maxAgeMillis: Long = 24 * 60 * 60 * 1000L) { // Default 24 hours
        val dir = File(cacheDir)
        if (dir.exists() && dir.isDirectory) {
            dir.listFiles()?.forEach { file ->
                if (System.currentTimeMillis() - file.lastModified() > maxAgeMillis) {
                    if (file.delete()) {
                        logDebug("Deleted expired cache file: ${file.name}")
                        hashToPathMap.entries.removeIf { it.value == file.absolutePath }
                    }
                }
            }
        }
    }

    companion object {
        private const val TAG = "MnnImageProcessor"
        private var instance: MnnImageProcessor? = null

        fun getInstance(context: Context): MnnImageProcessor {
            return instance ?: synchronized(this) {
                instance ?: MnnImageProcessor(context.applicationContext).also { instance = it }
            }
        }

        private fun logDebug(message: String) {
            Timber.tag(TAG).d(message)
        }

        private fun logError(message: String, throwable: Throwable? = null) {
            if (throwable != null) {
                Timber.tag(TAG).e(throwable, message)
            } else {
                Timber.tag(TAG).e(message)
            }
        }
    }
}
