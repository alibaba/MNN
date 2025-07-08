// Created by ruoyi.sjd on 2025/1/14.
// Copyright (c) 2024 Alibaba Group Holding Limited. All rights reserved.

package com.alibaba.mls.api

import android.content.Context
import android.content.SharedPreferences
import android.util.Log
import com.alibaba.mnnllm.android.modelmarket.ModelRepository
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch

object ModelTagsCache {
    private const val TAG = "ModelTagsCache"
    private const val PREFS_NAME = "model_tags_cache"
    private const val KEY_TAGS_CACHE = "tags_cache"
    private const val KEY_CACHE_VERSION = "cache_version"
    private const val CURRENT_CACHE_VERSION = 1

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val gson = Gson()
    
    // In-memory cache for quick access
    private var memoryCache: Map<String, List<String>> = emptyMap()
    private var cacheInitialized = false

    /**
     * Initialize the cache by loading model market data
     */
    fun initializeCache(context: Context) {
        if (cacheInitialized) return
        
        scope.launch {
            try {
                val prefs = getPrefs(context)
                val cacheVersion = prefs.getInt(KEY_CACHE_VERSION, 0)
                
                // Check if we need to refresh cache
                if (cacheVersion != CURRENT_CACHE_VERSION) {
                    Log.d(TAG, "Cache version mismatch, refreshing cache")
                    refreshCacheFromAssets(context)
                } else {
                    // Load from SharedPreferences
                    loadCacheFromPrefs(context)
                    
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
     * Refresh cache from assets
     */
    private suspend fun refreshCacheFromAssets(context: Context) {
        try {
            val repository = ModelRepository(context)
            val models = repository.getModels()
            
            val newCache = mutableMapOf<String, List<String>>()
            models.forEach { model ->
                newCache[model.modelName] = model.tags
            }
            
            // Update memory cache
            memoryCache = newCache
            
            // Save to SharedPreferences
            saveCacheToPrefs(context, newCache)
            
            Log.d(TAG, "Cache refreshed with ${newCache.size} entries")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to refresh cache from assets", e)
        }
    }

    /**
     * Load cache from SharedPreferences
     */
    private fun loadCacheFromPrefs(context: Context) {
        try {
            val prefs = getPrefs(context)
            val cacheJson = prefs.getString(KEY_TAGS_CACHE, null)
            
            if (cacheJson != null) {
                val type = object : TypeToken<Map<String, List<String>>>() {}.type
                memoryCache = gson.fromJson(cacheJson, type) ?: emptyMap()
                Log.d(TAG, "Loaded cache from prefs with ${memoryCache.size} entries")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load cache from prefs", e)
            memoryCache = emptyMap()
        }
    }

    /**
     * Save cache to SharedPreferences
     */
    private fun saveCacheToPrefs(context: Context, cache: Map<String, List<String>>) {
        try {
            val prefs = getPrefs(context)
            val cacheJson = gson.toJson(cache)
            
            prefs.edit()
                .putString(KEY_TAGS_CACHE, cacheJson)
                .putInt(KEY_CACHE_VERSION, CURRENT_CACHE_VERSION)
                .apply()
                
            Log.d(TAG, "Saved cache to prefs with ${cache.size} entries")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save cache to prefs", e)
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
     * Get SharedPreferences instance
     */
    private fun getPrefs(context: Context): SharedPreferences {
        return context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
    }

    /**
     * Clear cache (for debugging or cache invalidation)
     */
    fun clearCache(context: Context) {
        memoryCache = emptyMap()
        getPrefs(context).edit().clear().apply()
        cacheInitialized = false
        Log.d(TAG, "Cache cleared")
    }
} 