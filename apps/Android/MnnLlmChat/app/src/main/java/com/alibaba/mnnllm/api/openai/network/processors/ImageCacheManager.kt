package com.alibaba.mnnllm.api.openai.network.processors

import android.content.Context
import android.graphics.BitmapFactory
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import timber.log.Timber
import java.io.File
import java.security.MessageDigest
import java.util.Base64

/** * imagecachemanager，supportdualhashmechanism： * 1. stringhash（fast，1ms）- directlytoBase64stringcalculatehash * 2. contenthash（slow，tens to hundredsms）- todecodingafterimagedatacalculatehash * * solvedifferentclientBase64encodingdifferencescaused bycachemississue*/
class ImageCacheManager(private val context: Context) {
    
    @Serializable
    data class CacheEntry(
        val stringHash: String,      //Base64stringhash
        val contentHash: String,     //after decodingcontenthash
        val filePath: String,        //localfile path
        val timestamp: Long          //creation timestamp
    )
    
    @Serializable
    data class CacheIndex(
        val entries: MutableList<CacheEntry> = mutableListOf()
    )
    
    private val cacheDir: String by lazy {
        val dir = File(context.externalCacheDir, "mnn_image_cache")
        if (!dir.exists()) {
            dir.mkdirs()
        }
        dir.absolutePath
    }
    
    private val indexFile: File by lazy {
        File(cacheDir, "cache_index.json")
    }
    
    private var cacheIndex: CacheIndex = CacheIndex()
    private val json = Json { ignoreUnknownKeys = true }
    
    init {
        loadCacheIndex()
        cleanupInvalidEntries()
    }
    
    /** * process Base64image,usedual hashmechanism * @param base64Data Base64 encodingimage data * @return localfile path,ifprocessfails thenreturn null*/
    suspend fun processBase64Image(base64Data: String): String? = withContext(Dispatchers.IO) {
        try {
            logDebug("开始处理Base64图像，数据长度: ${base64Data.length}")
            
            if (base64Data.isEmpty()) {
                logError("Base64数据为空")
                return@withContext null
            }
            
            //Step 1: faststringhash check (about 1ms)
            val startTime = System.currentTimeMillis()
            val stringHash = calculateSHA256(base64Data)
            val stringHashTime = System.currentTimeMillis() - startTime
            logDebug("计算字符串哈希: $stringHash (耗时: ${stringHashTime}ms)")
            
            //checkstringhashwhetherhitcache
            val stringCacheHit = findByStringHash(stringHash)
            if (stringCacheHit != null && File(stringCacheHit.filePath).exists()) {
                logDebug("字符串哈希命中缓存: ${stringCacheHit.filePath}")
                return@withContext stringCacheHit.filePath
            }
            
            //Step 2: decoding image data (time-consuming)
            val decodeStartTime = System.currentTimeMillis()
            val imageData = try {
                Base64.getDecoder().decode(base64Data)
            } catch (e: IllegalArgumentException) {
                logError("Base64解码失败: ${e.message}")
                return@withContext null
            }
            val decodeTime = System.currentTimeMillis() - decodeStartTime
            logDebug("Base64解码完成，数据大小: ${imageData.size} bytes (耗时: ${decodeTime}ms)")
            
            //Step 3: calculatecontent hash (based on decodeddata)
            val contentHashStartTime = System.currentTimeMillis()
            val contentHash = calculateSHA256(imageData)
            val contentHashTime = System.currentTimeMillis() - contentHashStartTime
            logDebug("计算内容哈希: $contentHash (耗时: ${contentHashTime}ms)")
            
            //checkcontenthashwhetherhitcache
            val contentCacheHit = findByContentHash(contentHash)
            if (contentCacheHit != null && File(contentCacheHit.filePath).exists()) {
                logDebug("内容哈希命中缓存: ${contentCacheHit.filePath}")
                //updatecacheindex，addnewstringhashmap
                addStringHashMapping(stringHash, contentCacheHit)
                return@withContext contentCacheHit.filePath
            }
            
            //Step 4: save newfile
            val fileName = "img_${contentHash.substring(0, 8)}.jpg"
            val filePath = File(cacheDir, fileName).absolutePath
            
            val saveStartTime = System.currentTimeMillis()
            val savedPath = saveImageDataToFile(imageData, filePath)
            val saveTime = System.currentTimeMillis() - saveStartTime
            
            if (savedPath != null) {
                logDebug("保存图像文件成功: $savedPath (耗时: ${saveTime}ms)")
                
                //addtocacheindex
                val newEntry = CacheEntry(
                    stringHash = stringHash,
                    contentHash = contentHash,
                    filePath = savedPath,
                    timestamp = System.currentTimeMillis()
                )
                addCacheEntry(newEntry)
                
                val totalTime = System.currentTimeMillis() - startTime
                logDebug("Base64图像处理完成，总耗时: ${totalTime}ms")
                
                return@withContext savedPath
            } else {
                logError("保存图像文件失败")
                return@withContext null
            }
            
        } catch (e: Exception) {
            logError("处理Base64图像时发生错误: ${e.message}", e)
            null
        }
    }
    
    /** * according tostringhashfindcacheentry*/
    private fun findByStringHash(stringHash: String): CacheEntry? {
        return cacheIndex.entries.find { it.stringHash == stringHash }
    }
    
    /** * according tocontenthashfindcacheentry*/
    private fun findByContentHash(contentHash: String): CacheEntry? {
        return cacheIndex.entries.find { it.contentHash == contentHash }
    }
    
    /** * asexistingcacheentryaddnewstringhashmap*/
    private fun addStringHashMapping(stringHash: String, existingEntry: CacheEntry) {
        //check whetheralready existssame string hash
        if (findByStringHash(stringHash) == null) {
            val newEntry = existingEntry.copy(stringHash = stringHash)
            cacheIndex.entries.add(newEntry)
            saveCacheIndex()
            logDebug("添加字符串哈希映射: $stringHash -> ${existingEntry.filePath}")
        }
    }
    
    /** * addnewcacheentry*/
    private fun addCacheEntry(entry: CacheEntry) {
        cacheIndex.entries.add(entry)
        saveCacheIndex()
        logDebug("添加新缓存条目: ${entry.filePath}")
    }
    
    /** * saveimagedatatofile*/
    private suspend fun saveImageDataToFile(imageData: ByteArray, filePath: String): String? = 
        withContext(Dispatchers.IO) {
            val imageFile = File(filePath)
            try {
                imageFile.writeBytes(imageData)
                
                //verificationimagefilevalidity
                if (isValidImageFile(filePath)) {
                    filePath
                } else {
                    logError("保存的图像文件无效: $filePath")
                    imageFile.delete()
                    null
                }
            } catch (e: Exception) {
                logError("保存图像文件失败: ${e.message}", e)
                imageFile.delete()
                null
            }
        }
    
    /** * verificationimagefilevalidity*/
    private fun isValidImageFile(filePath: String): Boolean {
        return try {
            val options = BitmapFactory.Options()
            options.inJustDecodeBounds = true
            BitmapFactory.decodeFile(filePath, options)
            options.outWidth > 0 && options.outHeight > 0
        } catch (e: Exception) {
            logError("验证图像文件时发生错误: ${e.message}", e)
            false
        }
    }
    
    /** * calculatestringSHA256hash*/
    private fun calculateSHA256(input: String): String {
        return try {
            val digest = MessageDigest.getInstance("SHA-256")
            val hashBytes = digest.digest(input.toByteArray(Charsets.UTF_8))
            hashBytes.joinToString("") { "%02x".format(it) }
        } catch (e: Exception) {
            logError("计算SHA256哈希失败", e)
            input.hashCode().toString()
        }
    }
    
    /** * calculatebytearraySHA256hash*/
    private fun calculateSHA256(data: ByteArray): String {
        return try {
            val digest = MessageDigest.getInstance("SHA-256")
            val hashBytes = digest.digest(data)
            hashBytes.joinToString("") { "%02x".format(it) }
        } catch (e: Exception) {
            logError("计算SHA256哈希失败", e)
            data.contentHashCode().toString()
        }
    }
    
    /**
     * loadcacheindex*/
    private fun loadCacheIndex() {
        try {
            if (indexFile.exists()) {
                val jsonContent = indexFile.readText()
                cacheIndex = json.decodeFromString<CacheIndex>(jsonContent)
                logDebug("加载缓存索引成功，条目数量: ${cacheIndex.entries.size}")
            } else {
                logDebug("缓存索引文件不存在，创建新索引")
                cacheIndex = CacheIndex()
            }
        } catch (e: Exception) {
            logError("加载缓存索引失败: ${e.message}", e)
            cacheIndex = CacheIndex()
        }
    }
    
    /**
     * savecacheindex*/
    private fun saveCacheIndex() {
        try {
            val jsonContent = json.encodeToString(CacheIndex.serializer(), cacheIndex)
            indexFile.writeText(jsonContent)
            logDebug("保存缓存索引成功")
        } catch (e: Exception) {
            logError("保存缓存索引失败: ${e.message}", e)
        }
    }
    
    /** * cleanupinvalidcacheentry（filenon-existententry）*/
    private fun cleanupInvalidEntries() {
        val initialSize = cacheIndex.entries.size
        cacheIndex.entries.removeAll { entry ->
            val fileExists = File(entry.filePath).exists()
            if (!fileExists) {
                logDebug("清理无效缓存条目: ${entry.filePath}")
            }
            !fileExists
        }
        
        val removedCount = initialSize - cacheIndex.entries.size
        if (removedCount > 0) {
            saveCacheIndex()
            logDebug("清理完成，移除了 $removedCount 个无效条目")
        }
    }
    
    /** * cleanupexpiredcache*/
    fun cleanupExpiredCache(maxAgeMillis: Long = 24 * 60 * 60 * 1000L) {
        val currentTime = System.currentTimeMillis()
        val initialSize = cacheIndex.entries.size
        
        cacheIndex.entries.removeAll { entry ->
            val isExpired = currentTime - entry.timestamp > maxAgeMillis
            if (isExpired) {
                val file = File(entry.filePath)
                if (file.exists()) {
                    file.delete()
                }
                logDebug("清理过期缓存: ${entry.filePath}")
            }
            isExpired
        }
        
        val removedCount = initialSize - cacheIndex.entries.size
        if (removedCount > 0) {
            saveCacheIndex()
            logDebug("清理过期缓存完成，移除了 $removedCount 个条目")
        }
    }
    
    /** * getcachestatisticsinfo*/
    fun getCacheStats(): String {
        val totalEntries = cacheIndex.entries.size
        val validFiles = cacheIndex.entries.count { File(it.filePath).exists() }
        val totalSize = cacheIndex.entries
            .filter { File(it.filePath).exists() }
            .sumOf { File(it.filePath).length() }
        
        return "缓存统计: 总条目=$totalEntries, 有效文件=$validFiles, 总大小=${totalSize / 1024}KB"
    }
    
    companion object {
        private const val TAG = "ImageCacheManager"
        private var instance: ImageCacheManager? = null
        
        fun getInstance(context: Context): ImageCacheManager {
            return instance ?: synchronized(this) {
                instance ?: ImageCacheManager(context.applicationContext).also { instance = it }
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