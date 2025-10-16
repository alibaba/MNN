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

/** * MNN imageprocessor,responsible for processingimage URL (Base64 ornetworkURL）, * will andsave tolocalcache andreturn file path. * integratedual hashcachemechanismto solvedifferentclientBase64 encodingdifferencesissue.*/
class MnnImageProcessor(private val context: Context) {
    private val hashToPathMap: MutableMap<String, String> = HashMap()
    private val imageCacheManager = ImageCacheManager.getInstance(context)
    private val cacheDir: String by lazy {
        val dir = File(context.externalCacheDir, "mnn_image_cache")
        if (!dir.exists()) {
            dir.mkdirs()
        }
        dir.absolutePath
    }

    init {
        ensureCacheDirExists()
        //start whencleanupinvalidcacheentry
        logDebug("初始化图像处理器，清理缓存...")
        logDebug(imageCacheManager.getCacheStats())
    }

    private fun ensureCacheDirExists() {
        val dir = File(cacheDir)
        if (!dir.exists()) {
            if (!dir.mkdirs()) {
                logError("Failed to create cache directory: $cacheDir")
            }
        }
    }

    /** * process imageURL,possiblyis Base64data URI ornetworkURL. * @param imageUrl image URL orBase64 data URI * @return localfile path,ifprocessfails thenreturn null*/
    suspend fun processImageUrl(imageUrl: String): String? {
        logDebug("Starting to process image URL. Input type detection...")
        return if (imageUrl.startsWith("data:image")) {
            logDebug("Detected Base64 data URI. Parsing...")
            val parts = imageUrl.split(",")
            if (parts.size == 2) {
                logDebug("Base64 data parsed successfully. Header: ${parts[0]}, Data length: ${parts[1].length}")
                processBase64Image(parts[1])
            } else {
                logError("Invalid Base64 data URI format. Expected 2 parts but got ${parts.size}. Full input: $imageUrl")
                null
            }
        } else if (imageUrl.startsWith("http://") || imageUrl.startsWith("https://")) {
            //networkURL
            logDebug("Processing network image URL: $imageUrl")
            //TODO: implementationnetworkimagedownloadandcachelogic
            //temporarily returnoriginal URLor aplaceholder,ordownload it
            downloadAndProcessNetworkImage(imageUrl)
        } else if (File(imageUrl).exists()) {
            //already islocalfile path
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

                //use URLhash orfile nameascache key
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

    private suspend fun processBase64Image(base64Data: String): String? {
        logDebug("使用双重哈希缓存机制处理Base64图像")
        
        //usenewImageCacheManagerprocessBase64image
        val result = imageCacheManager.processBase64Image(base64Data)
        
        if (result != null) {
            logDebug("Base64图像处理成功: $result")
            return result
        } else {
            logError("Base64图像处理失败，回退到原有方法")
            //fallbacktooriginalprocessmethod
            return processBase64ImageFallback(base64Data)
        }
    }
    
    /** * original Base64processing method,asfallbacksolution*/
    private suspend fun processBase64ImageFallback(base64Data: String): String? =
        withContext(Dispatchers.IO) {
            try {
                logDebug("Starting Base64 image processing (fallback). Data length: ${base64Data.length}")
                if (base64Data.isEmpty()) {
                    logError("Invalid base64 data: empty input")
                    return@withContext null
                }

                val startTime = System.currentTimeMillis()
                val hash = calculateSHA256(base64Data)
                val hashTime = System.currentTimeMillis() - startTime
                logDebug("Calculated SHA-256 hash: $hash (took ${hashTime}ms). First 16 chars: ${base64Data.take(16)}...")

                if (hashToPathMap.containsKey(hash)) {
                    val cachedPath = hashToPathMap[hash]
                    if (cachedPath != null && File(cachedPath).exists()) {
                        logDebug("Found cached base64 image: $cachedPath")
                        return@withContext cachedPath
                    }
                }

                val decodeStart = System.currentTimeMillis()
                val imageData = Base64.getDecoder().decode(base64Data)
                val decodeTime = System.currentTimeMillis() - decodeStart
                logDebug("Decoded Base64 data. Raw bytes: ${imageData.size} (took ${decodeTime}ms)")

                val fileName = "img_" + hash.substring(0, 8) + ".jpg" // Assume jpg for base64
                val filePath = File(cacheDir, fileName).absolutePath
                logDebug("Generating cache file path. Directory: $cacheDir, File name: $fileName")

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
            logDebug("Attempting to save image data to $filePath. Data size: ${imageData.size} bytes")
            val startTime = System.currentTimeMillis()
            FileOutputStream(imageFile).use { fos ->
                fos.write(imageData)
            }
            val writeTime = System.currentTimeMillis() - startTime
            logDebug("Successfully saved image file: $filePath (took ${writeTime}ms)")

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
        logDebug("Validating image file: $filePath")
        return try {
            val options = BitmapFactory.Options()
            options.inJustDecodeBounds = true
            val startTime = System.currentTimeMillis()
            BitmapFactory.decodeFile(filePath, options)
            val decodeTime = System.currentTimeMillis() - startTime
            val isValid = options.outWidth > 0 && options.outHeight > 0
            logDebug("Image validation result: $isValid. Dimensions: ${options.outWidth}x${options.outHeight} (took ${decodeTime}ms)")
            isValid
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

    /** * cleanup expiredcache (includingnewdual hashcache andold cache)*/
    fun cleanupCache(maxAgeMillis: Long = 24 * 60 * 60 * 1000L) { // Default 24 hours
        //cleanupnewdualhashcache
        imageCacheManager.cleanupExpiredCache(maxAgeMillis)
        
        //cleanup oldcache method (backward compatible)
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
    
    /** * getcachestatisticsinfo*/
    fun getCacheStats(): String {
        return imageCacheManager.getCacheStats()
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
