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

/**
 * 图像缓存管理器，支持双重哈希机制：
 * 1. 字符串哈希（快速，1ms）- 直接对Base64字符串计算哈希
 * 2. 内容哈希（慢速，几十到几百ms）- 对解码后的图像数据计算哈希
 * 
 * 解决不同客户端Base64编码差异导致的缓存未命中问题
 */
class ImageCacheManager(private val context: Context) {
    
    @Serializable
    data class CacheEntry(
        val stringHash: String,      // Base64字符串的哈希
        val contentHash: String,     // 解码后内容的哈希
        val filePath: String,        // 本地文件路径
        val timestamp: Long          // 创建时间戳
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
    
    /**
     * 处理Base64图像，使用双重哈希机制
     * @param base64Data Base64编码的图像数据
     * @return 本地文件路径，如果处理失败则返回null
     */
    suspend fun processBase64Image(base64Data: String): String? = withContext(Dispatchers.IO) {
        try {
            logDebug("开始处理Base64图像，数据长度: ${base64Data.length}")
            
            if (base64Data.isEmpty()) {
                logError("Base64数据为空")
                return@withContext null
            }
            
            // 第一步：快速字符串哈希检查（约1ms）
            val startTime = System.currentTimeMillis()
            val stringHash = calculateSHA256(base64Data)
            val stringHashTime = System.currentTimeMillis() - startTime
            logDebug("计算字符串哈希: $stringHash (耗时: ${stringHashTime}ms)")
            
            // 检查字符串哈希是否命中缓存
            val stringCacheHit = findByStringHash(stringHash)
            if (stringCacheHit != null && File(stringCacheHit.filePath).exists()) {
                logDebug("字符串哈希命中缓存: ${stringCacheHit.filePath}")
                return@withContext stringCacheHit.filePath
            }
            
            // 第二步：解码图像数据（耗时较长）
            val decodeStartTime = System.currentTimeMillis()
            val imageData = try {
                Base64.getDecoder().decode(base64Data)
            } catch (e: IllegalArgumentException) {
                logError("Base64解码失败: ${e.message}")
                return@withContext null
            }
            val decodeTime = System.currentTimeMillis() - decodeStartTime
            logDebug("Base64解码完成，数据大小: ${imageData.size} bytes (耗时: ${decodeTime}ms)")
            
            // 第三步：计算内容哈希（基于解码后的数据）
            val contentHashStartTime = System.currentTimeMillis()
            val contentHash = calculateSHA256(imageData)
            val contentHashTime = System.currentTimeMillis() - contentHashStartTime
            logDebug("计算内容哈希: $contentHash (耗时: ${contentHashTime}ms)")
            
            // 检查内容哈希是否命中缓存
            val contentCacheHit = findByContentHash(contentHash)
            if (contentCacheHit != null && File(contentCacheHit.filePath).exists()) {
                logDebug("内容哈希命中缓存: ${contentCacheHit.filePath}")
                // 更新缓存索引，添加新的字符串哈希映射
                addStringHashMapping(stringHash, contentCacheHit)
                return@withContext contentCacheHit.filePath
            }
            
            // 第四步：保存新文件
            val fileName = "img_${contentHash.substring(0, 8)}.jpg"
            val filePath = File(cacheDir, fileName).absolutePath
            
            val saveStartTime = System.currentTimeMillis()
            val savedPath = saveImageDataToFile(imageData, filePath)
            val saveTime = System.currentTimeMillis() - saveStartTime
            
            if (savedPath != null) {
                logDebug("保存图像文件成功: $savedPath (耗时: ${saveTime}ms)")
                
                // 添加到缓存索引
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
    
    /**
     * 根据字符串哈希查找缓存条目
     */
    private fun findByStringHash(stringHash: String): CacheEntry? {
        return cacheIndex.entries.find { it.stringHash == stringHash }
    }
    
    /**
     * 根据内容哈希查找缓存条目
     */
    private fun findByContentHash(contentHash: String): CacheEntry? {
        return cacheIndex.entries.find { it.contentHash == contentHash }
    }
    
    /**
     * 为已存在的缓存条目添加新的字符串哈希映射
     */
    private fun addStringHashMapping(stringHash: String, existingEntry: CacheEntry) {
        // 检查是否已存在相同的字符串哈希
        if (findByStringHash(stringHash) == null) {
            val newEntry = existingEntry.copy(stringHash = stringHash)
            cacheIndex.entries.add(newEntry)
            saveCacheIndex()
            logDebug("添加字符串哈希映射: $stringHash -> ${existingEntry.filePath}")
        }
    }
    
    /**
     * 添加新的缓存条目
     */
    private fun addCacheEntry(entry: CacheEntry) {
        cacheIndex.entries.add(entry)
        saveCacheIndex()
        logDebug("添加新缓存条目: ${entry.filePath}")
    }
    
    /**
     * 保存图像数据到文件
     */
    private suspend fun saveImageDataToFile(imageData: ByteArray, filePath: String): String? = 
        withContext(Dispatchers.IO) {
            val imageFile = File(filePath)
            try {
                imageFile.writeBytes(imageData)
                
                // 验证图像文件有效性
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
    
    /**
     * 验证图像文件有效性
     */
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
    
    /**
     * 计算字符串的SHA256哈希
     */
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
    
    /**
     * 计算字节数组的SHA256哈希
     */
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
     * 加载缓存索引
     */
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
     * 保存缓存索引
     */
    private fun saveCacheIndex() {
        try {
            val jsonContent = json.encodeToString(CacheIndex.serializer(), cacheIndex)
            indexFile.writeText(jsonContent)
            logDebug("保存缓存索引成功")
        } catch (e: Exception) {
            logError("保存缓存索引失败: ${e.message}", e)
        }
    }
    
    /**
     * 清理无效的缓存条目（文件不存在的条目）
     */
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
    
    /**
     * 清理过期缓存
     */
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
    
    /**
     * 获取缓存统计信息
     */
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