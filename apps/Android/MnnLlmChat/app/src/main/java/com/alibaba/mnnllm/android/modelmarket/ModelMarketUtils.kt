// Created by ruoyi.sjd on 2025/7/17.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.modelmarket

import android.content.Context
import android.util.Log
import com.alibaba.mls.api.ApplicationProvider
import com.alibaba.mnnllm.android.modelsettings.ModelConfig
import com.google.gson.Gson
import com.google.gson.GsonBuilder
import com.google.gson.JsonParser
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.util.concurrent.ConcurrentHashMap

object ModelMarketUtils {

    const val TAG = "ModelMarketUtils"
    
    // Cache all model market data in memory
    private val marketItemsCache = ConcurrentHashMap<String, ModelMarketItem>()
    private var isInitialized = false
    private var currentMarketVersion: String? = null

    /**
     * Initialize memory cache by loading all model data from ModelRepository
     */
    suspend fun initializeCache(context: Context) = withContext(Dispatchers.IO) {
        if (isInitialized) {
            Log.d(TAG, "Cache already initialized")
            return@withContext
        }
        
        Log.d(TAG, "Initializing ModelMarketUtils cache...")
        val startTime = System.currentTimeMillis()
        
        try {
            val repository = ModelRepository(context)
            
            // Get market data version
            val marketData = repository.getModelMarketData()
            currentMarketVersion = marketData?.version
            
            // Clear cache and reload
            marketItemsCache.clear()
            
            // Load all types of models into cache
            val allModels = repository.getModels()
            val ttsModels = repository.getTtsModels()
            val asrModels = repository.getAsrModels()
            
            // Cache all models
            allModels.forEach { marketItemsCache[it.modelId] = it }
            ttsModels.forEach { marketItemsCache[it.modelId] = it }
            asrModels.forEach { marketItemsCache[it.modelId] = it }
            
            // Load local and built-in model configurations
            loadLocalAndBuiltinModels(context)
            
            isInitialized = true
            val duration = System.currentTimeMillis() - startTime
            Log.d(TAG, "ModelMarketUtils cache initialized in ${duration}ms with ${marketItemsCache.size} models")
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize ModelMarketUtils cache", e)
        }
    }
    
    /**
     * Load local and built-in model configurations into cache
     */
    private suspend fun loadLocalAndBuiltinModels(context: Context) = withContext(Dispatchers.IO) {
        try {
            // Scan local config files under .mnnmodels directory
            val mnnModelsDir = File(context.filesDir, ".mnnmodels")
            if (mnnModelsDir.exists()) {
                scanForLocalConfigs(mnnModelsDir)
            }
            
            Log.d(TAG, "Loaded local and builtin model configs")
        } catch (e: Exception) {
            Log.w(TAG, "Failed to load local and builtin models", e)
        }
    }
    
    /**
     * Recursively scan directory to find and load market_config.json files
     */
    private fun scanForLocalConfigs(directory: File) {
        try {
            directory.listFiles()?.forEach { file ->
                if (file.isDirectory) {
                    scanForLocalConfigs(file)
                } else if (file.name == "market_config.json") {
                    try {
                        val configJson = file.readText()
                        val marketItem = Gson().fromJson(configJson, ModelMarketItem::class.java)
                        if (marketItem?.modelId != null) {
                            marketItemsCache[marketItem.modelId] = marketItem
                            Log.d(TAG, "Loaded local config for ${marketItem.modelId}")
                        }
                    } catch (e: Exception) {
                        Log.w(TAG, "Failed to load config from ${file.absolutePath}", e)
                    }
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to scan directory ${directory.absolutePath}", e)
        }
    }
    
    /**
     * Get all cached model data
     */
    fun getAllCachedModels(): Map<String, ModelMarketItem> {
        return marketItemsCache.toMap()
    }
    
    /**
     * Get model data by modelId (from memory cache)
     */
    fun getModelFromCache(modelId: String): ModelMarketItem? {
        return marketItemsCache[modelId]
    }
    
    /**
     * Check if cache is initialized
     */
    fun isCacheInitialized(): Boolean {
        return isInitialized
    }
    
    /**
     * Force refresh cache
     */
    suspend fun refreshCache(context: Context) = withContext(Dispatchers.IO) {
        isInitialized = false
        initializeCache(context)
    }

    fun writeMarketConfig(modelItem: ModelMarketItem, marketVersion: String? = null) {
        // Update memory cache simultaneously
        marketItemsCache[modelItem.modelId] = modelItem
        
        // Create a map with all fields except 'sources', and add 'modelId'
        val configMap = mutableMapOf<String, Any?>()
        configMap["modelName"] = modelItem.modelName
        configMap["vendor"] = modelItem.vendor
        configMap["size_gb"] = modelItem.sizeB
        configMap["tags"] = modelItem.tags
        configMap["extra_tags"] = modelItem.extraTags
        configMap["categories"] = modelItem.categories
        configMap["description"] = modelItem.description
        configMap["modelId"] = modelItem.modelId
        // Pretty-print JSON
        val gson = GsonBuilder().setPrettyPrinting().create()
        val jsonString = gson.toJson(configMap)
        // Write to file
        val filePath = ModelConfig.getMarketConfigFile(modelItem.modelId)
        try {
            val file = File(filePath)
            file.parentFile?.mkdirs()
            file.writeText(jsonString)
            Log.d(TAG, "Wrote market config to $filePath")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to write market config for ${modelItem.modelId}", e)
        }
    }


    suspend fun readMarketConfig(modelId: String): ModelMarketItem? {
        Log.d(TAG, "readMarketConfig for $modelId")
        
        // Ensure cache is initialized
        val context = ApplicationProvider.get()
        if (!isInitialized) {
            initializeCache(context)
        }
        
        // Try getting from memory cache first
        val cachedItem = marketItemsCache[modelId]
        if (cachedItem != null) {
            Log.d(TAG, "Found $modelId in memory cache")
            return cachedItem
        }
        
        // If not in cache, try reading from local file (fallback)
        Log.d(TAG, "Model $modelId not found in cache, trying local file")
        return readMarketConfigFromLocal(modelId)
    }

    suspend fun readMarketConfigFromLocal(modelId: String): ModelMarketItem? {
        return withContext(Dispatchers.IO) {
            val marketConfigFile = ModelConfig.getMarketConfigFile(modelId)
            Log.d(TAG, "readMarketConfigFromLocal for $modelId: looking for file at $marketConfigFile")
            var marketItem: ModelMarketItem? = null
            try {
                val configFile = File(marketConfigFile)
                Log.d(TAG, "Config file exists: ${configFile.exists()}")
                if (configFile.exists()) {
                    val configJson = configFile.readText()
                    Log.d(TAG, "Config file content: $configJson")
                    marketItem = Gson().fromJson(configJson, ModelMarketItem::class.java)
                    Log.d(TAG, "Loaded market config for ${modelId}: ${marketItem?.modelName}, tags=${marketItem?.tags}")
                    
                    // If successfully read, update memory cache as well
                    if (marketItem != null) {
                        marketItemsCache[modelId] = marketItem
                    }
                } else {
                    Log.w(TAG, "Market config file does not exist: $marketConfigFile")
                }
            } catch (e: Exception) {
                Log.w(TAG, "Failed to load market config for ${modelId}", e)
            }
            marketItem
        }
    }
}