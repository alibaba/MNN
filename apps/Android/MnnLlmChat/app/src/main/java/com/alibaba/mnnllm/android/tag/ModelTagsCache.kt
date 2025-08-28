// Created by ruoyi.sjd on 2025/1/14.
// Copyright (c) 2024 Alibaba Group Holding Limited. All rights reserved.

package com.alibaba.mnnllm.android.tag

import android.content.Context
import android.util.Log
import com.alibaba.mnnllm.android.modelmarket.ModelRepository
import com.alibaba.mnnllm.android.modelsettings.ModelConfig
import com.google.gson.Gson
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import java.io.File

object ModelTagsCache {
    private const val TAG = "ModelTagsCache"
    private const val CACHE_FILE_NAME = "model_tags_cache.json"
    private const val CURRENT_CACHE_VERSION = 1

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val gson = Gson()
    
    // In-memory cache for quick access
    private var memoryCache: Map<String, List<String>> = emptyMap()
    private var tagTranslationsCache: Map<String, String> = emptyMap()
    private var cacheInitialized = false
    
    // Cache data structure
    private data class CacheData(
        val version: Int,
        val tags: Map<String, List<String>>,
        val tagTranslations: Map<String, String> = emptyMap()
    )

    /**
     * Initialize the cache by loading model market data
     */
    fun initializeCache(context: Context) {
        if (cacheInitialized) return
        
        scope.launch {
            try {
                val cacheFile = getCacheFile(context)
                val cacheData = loadCacheFromFile(cacheFile)
                
                // Check if we need to refresh cache
                if (cacheData == null || cacheData.version != CURRENT_CACHE_VERSION) {
                    Log.d(TAG, "Cache version mismatch or not found, refreshing cache")
                    refreshCacheFromAssets(context)
                } else {
                    // Load from file
                    memoryCache = cacheData.tags
                    tagTranslationsCache = cacheData.tagTranslations
                    
                    // If memory cache is empty, refresh from assets
                    if (memoryCache.isEmpty()) {
                        Log.d(TAG, "Memory cache is empty, refreshing from assets")
                        refreshCacheFromAssets(context)
                    }
                }
                
                cacheInitialized = true
                Log.d(TAG, "Cache initialized with ${memoryCache.size} entries")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to initialize cache", e)
                cacheInitialized = true // Prevent infinite retry
            }
        }
    }

    /**
     * Get tags for a model by modelId
     */
    fun getTagsForModel(context: Context, modelId: String): List<String> {
        if (!cacheInitialized) {
            initializeCache(context)
        }
        
        // Extract model name from modelId for matching
        val modelName = extractModelNameFromId(modelId)
        
        // Check memory cache first
        val cachedTags = memoryCache[modelName]
        if (cachedTags != null) {
            Log.d(TAG, "Found cached tags for $modelName: $cachedTags")
            return cachedTags
        }
        
        Log.d(TAG, "No cached tags found for $modelName")
        return emptyList()
    }

    /**
     * Get Chinese translation for a tag
     */
    fun getTagTranslation(context: Context, tag: String): String {
        if (!cacheInitialized) {
            initializeCache(context)
        }
        
        // Return Chinese translation if available, otherwise return original tag
        return tagTranslationsCache[tag] ?: tag
    }

    /**
     * Get Chinese translations for multiple tags
     */
    fun getTagTranslations(context: Context, tags: List<String>): List<String> {
        if (!cacheInitialized) {
            initializeCache(context)
        }
        
        return tags.map { tag ->
            tagTranslationsCache[tag] ?: tag
        }
    }

    /**
     * Refresh cache from assets
     */
    private suspend fun refreshCacheFromAssets(context: Context) {
        try {
            val repository = ModelRepository(context)
            val models = repository.getModels()
            val modelMarketData = repository.getModelMarketData()
            
            val newCache = mutableMapOf<String, List<String>>()
            models.forEach { model ->
                newCache[model.modelName] = model.tags
            }
            
            // Get tag translations from model market data
            val tagTranslations = modelMarketData?.tagTranslations ?: emptyMap()
            
            // Update memory cache
            memoryCache = newCache
            tagTranslationsCache = tagTranslations
            
            // Save to file
            saveCacheToFile(context, newCache, tagTranslations)
            
            Log.d(TAG, "Cache refreshed with ${newCache.size} entries and ${tagTranslations.size} translations")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to refresh cache from assets", e)
        }
    }

    /**
     * Get cache file path using ModelConfig directory
     */
    private fun getCacheFile(context: Context): File {
        val configDir = ModelConfig.getModelConfigDir("global")
        val dir = File(configDir)
        if (!dir.exists()) {
            dir.mkdirs()
        }
        return File(dir, CACHE_FILE_NAME)
    }

    /**
     * Load cache from file
     */
    private fun loadCacheFromFile(cacheFile: File): CacheData? {
        return try {
            if (!cacheFile.exists()) {
                Log.d(TAG, "Cache file does not exist")
                return null
            }
            
            val cacheJson = cacheFile.readText()
            val cacheData = gson.fromJson(cacheJson, CacheData::class.java)
            Log.d(TAG, "Loaded cache from file with ${cacheData.tags.size} entries")
            cacheData
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load cache from file", e)
            null
        }
    }

    /**
     * Save cache to file
     */
    private fun saveCacheToFile(context: Context, cache: Map<String, List<String>>, tagTranslations: Map<String, String>) {
        try {
            val cacheFile = getCacheFile(context)
            val cacheData = CacheData(CURRENT_CACHE_VERSION, cache, tagTranslations)
            val cacheJson = gson.toJson(cacheData)
            
            cacheFile.writeText(cacheJson)
            Log.d(TAG, "Saved cache to file with ${cache.size} entries and ${tagTranslations.size} translations")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save cache to file", e)
        }
    }

    /**
     * Extract model name from modelId for cache lookup
     * e.g., "ModelScope/MNN/Qwen3-4B-MNN" -> "Qwen3-4B-MNN"
     */
    private fun extractModelNameFromId(modelId: String): String {
        return when {
            modelId.contains("/") -> modelId.substringAfterLast("/")
            else -> modelId
        }
    }

    /**
     * Clear cache (for debugging or cache invalidation)
     */
    fun clearCache(context: Context) {
        memoryCache = emptyMap()
        tagTranslationsCache = emptyMap()
        try {
            val cacheFile = getCacheFile(context)
            if (cacheFile.exists()) {
                cacheFile.delete()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to delete cache file", e)
        }
        cacheInitialized = false
        Log.d(TAG, "Cache cleared")
    }
} 