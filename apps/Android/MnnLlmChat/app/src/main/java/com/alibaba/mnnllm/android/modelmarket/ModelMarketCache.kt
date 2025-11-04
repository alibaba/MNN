// Created by ruoyi.sjd on 2025/10/22.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.modelmarket

import android.content.Context
import com.alibaba.mls.api.ApplicationProvider
import com.alibaba.mnnllm.android.modelsettings.ModelConfig
import com.google.gson.Gson
import com.google.gson.GsonBuilder
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.filterIsInstance
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import timber.log.Timber
import java.io.File
import java.util.concurrent.ConcurrentHashMap

/**
 * ModelMarketCache is responsible for:
 * 1. Subscribing to ModelRepository changes
 * 2. Writing changed market configs to local disk
 * 3. Providing cached model data access
 * 4. Loading local and built-in model configurations
 */
object ModelMarketCache {

    private const val TAG = "ModelMarketCache"
    
    // Application scope for automatic initialization and subscription
    private val cacheScope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    
    // Memory cache for all model market data
    private val marketItemsCache = ConcurrentHashMap<String, ModelMarketItem>()
    
    // State management
    private val _cacheState = MutableStateFlow<CacheState>(CacheState.Uninitialized)
    val cacheState: StateFlow<CacheState> = _cacheState.asStateFlow()
    
    private var isInitialized = false
    private val initMutex = Mutex()
    private var currentMarketVersion: String? = null
    
    // Track previous data to detect changes
    private var previousMarketData: ModelMarketConfig? = null
    
    sealed class CacheState {
        object Uninitialized : CacheState()
        object Initializing : CacheState()
        data class Ready(val modelCount: Int) : CacheState()
        data class Error(val exception: Throwable) : CacheState()
    }

    /**
     * Initialize the cache and start subscribing to ModelRepository changes
     */
    suspend fun initialize(context: Context) = initMutex.withLock {
        if (isInitialized) {
            Timber.d("Cache already initialized")
            return
        }
        
        _cacheState.value = CacheState.Initializing
        
        try {
            val flow = ModelRepository.subscribeToMarketData()
            val initialData = flow.first()
            loadMarketDataToCache(initialData)
            previousMarketData = initialData
            
            // Load local and built-in model configurations
            loadLocalAndBuiltinModels(context)
            
            // Start subscribing to ModelRepository changes
            startSubscription(context)
            
            isInitialized = true
            val modelCount = marketItemsCache.size
            _cacheState.value = CacheState.Ready(modelCount)
            Timber.d("ModelMarketCache initialized with $modelCount models")
            
        } catch (e: Exception) {
            Timber.e(e, "Failed to initialize ModelMarketCache")
            _cacheState.value = CacheState.Error(e)
        }
    }
    
    /**
     * Subscribe to ModelRepository market data changes
     */
    private fun startSubscription(context: Context) {
        cacheScope.launch {
            ModelRepository.marketDataFlow.collect { newMarketData ->
                if (newMarketData == null) return@collect
                
                // Check if data actually changed
                val hasChanges = detectChanges(previousMarketData, newMarketData)
                if (hasChanges.isNotEmpty()) {
                    Timber.d("Detected ${hasChanges.size} model changes from ModelRepository")
                    onMarketDataChanged(context, newMarketData, hasChanges)
                    previousMarketData = newMarketData
                }
            }
        }
    }
    
    /**
     * Detect which models have changed between old and new data
     * Returns a set of modelIds that have changed
     */
    private fun detectChanges(
        oldData: ModelMarketConfig?,
        newData: ModelMarketConfig
    ): Set<String> {
        val changedModels = mutableSetOf<String>()
        
        // If no previous data, all models are new
        if (oldData == null) {
            return (newData.llmModels + newData.ttsModels + newData.asrModels + newData.libs)
                .map { it.modelId }
                .toSet()
        }
        
        // Build old and new model maps for comparison
        val oldModels = buildModelMap(oldData)
        val newModels = buildModelMap(newData)
        
        // Check for added or modified models
        newModels.forEach { (modelId, newItem) ->
            val oldItem = oldModels[modelId]
            if (oldItem == null || hasModelChanged(oldItem, newItem)) {
                changedModels.add(modelId)
            }
        }
        
        // Note: We don't track deleted models since they don't need cache updates
        
        return changedModels
    }
    
    /**
     * Build a map of modelId to ModelMarketItem from config
     */
    private fun buildModelMap(config: ModelMarketConfig): Map<String, ModelMarketItem> {
        val map = mutableMapOf<String, ModelMarketItem>()
        config.llmModels.forEach { map[it.modelId] = it }
        config.ttsModels.forEach { map[it.modelId] = it }
        config.asrModels.forEach { map[it.modelId] = it }
        config.libs.forEach { map[it.modelId] = it }
        return map
    }
    
    /**
     * Check if a model has changed
     */
    private fun hasModelChanged(old: ModelMarketItem, new: ModelMarketItem): Boolean {
        return old.modelName != new.modelName ||
               old.vendor != new.vendor ||
               old.tags != new.tags ||
               old.extraTags != new.extraTags ||
               old.categories != new.categories ||
               old.description != new.description ||
               old.sizeB != new.sizeB
    }
    
    /**
     * Handle market data changes by updating cache and writing to disk
     */
    private suspend fun onMarketDataChanged(
        context: Context,
        newData: ModelMarketConfig,
        changedModelIds: Set<String>
    ) = withContext(Dispatchers.IO) {
        try {
            // Update memory cache
            loadMarketDataToCache(newData)
            currentMarketVersion = newData.version
            
            // Write changed models to disk
            val newModels = buildModelMap(newData)
            changedModelIds.forEach { modelId ->
                val modelItem = newModels[modelId]
                if (modelItem != null) {
                    writeMarketConfigToLocal(modelItem, newData.version)
                }
            }
            
            _cacheState.value = CacheState.Ready(marketItemsCache.size)
            
        } catch (e: Exception) {
            Timber.e(e, "Failed to handle market data changes")
        }
    }
    
    /**
     * Load market data into memory cache
     */
    private fun loadMarketDataToCache(config: ModelMarketConfig) {
        marketItemsCache.clear()
        config.llmModels.forEach { marketItemsCache[it.modelId] = it }
        config.ttsModels.forEach { marketItemsCache[it.modelId] = it }
        config.asrModels.forEach { marketItemsCache[it.modelId] = it }
        config.libs.forEach { marketItemsCache[it.modelId] = it }
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
        } catch (e: Exception) {
            Timber.w(e, "Failed to load local and builtin models")
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
                        }
                    } catch (e: Exception) {
                        Timber.w(e, "Failed to load config from ${file.absolutePath}")
                    }
                }
            }
        } catch (e: Exception) {
            Timber.w(e, "Failed to scan directory ${directory.absolutePath}")
        }
    }
    
    /**
     * Write market config to local disk
     */
    private fun writeMarketConfigToLocal(modelItem: ModelMarketItem, marketVersion: String?) {
        try {
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
            val file = File(filePath)
            file.parentFile?.mkdirs()
            file.writeText(jsonString)
        } catch (e: Exception) {
            Timber.e(e, "Failed to write market config for ${modelItem.modelId}")
        }
    }
    
    // ==================== PUBLIC API ====================
    
    /**
     * Get model data by modelId (from memory cache)
     * Ensures cache is initialized before returning data
     */
    suspend fun getModelFromCache(modelId: String): ModelMarketItem? {
        // Ensure cache is initialized
        val context = ApplicationProvider.get()
        if (!isInitialized) {
            initialize(context)
        }
        
        // Wait for cache to be ready
        cacheState
            .filterIsInstance<CacheState.Ready>()
            .first()
        
        return marketItemsCache[modelId]
    }
    
    /**
     * Get all cached model data
     */
    fun getAllCachedModels(): Map<String, ModelMarketItem> {
        return marketItemsCache.toMap()
    }
    
    /**
     * Check if cache is initialized
     */
    fun isCacheInitialized(): Boolean {
        return isInitialized
    }
    
    /**
     * Get cache state as Flow for observation

     */
    fun observeCacheState(): StateFlow<CacheState> {
        return cacheState
    }
    
    /**
     * Observe model market config as Flow
     * Automatically initializes cache if needed
     * Emits Map<String, ModelMarketItem> when cache is ready
     */
    fun observeModelMarketConfig(context: Context): Flow<Map<String, ModelMarketItem>> = flow {
        // Auto-initialize if not already initialized
        if (!isInitialized) {
            initialize(context)
        }
        
        // Wait for cache to be ready and emit the data
        cacheState
            .filterIsInstance<CacheState.Ready>()
            .map { marketItemsCache.toMap() }
            .collect { emit(it) }
    }
    
    /**
     * Force refresh cache from ModelRepository
     */
    suspend fun refreshCache(context: Context) = withContext(Dispatchers.IO) {
        try {
            Timber.d("Force refreshing ModelMarketCache...")
            
            // Trigger ModelRepository refresh
            ModelRepository.forceRefresh()
            
            // The subscription will automatically handle the update
        } catch (e: Exception) {
            Timber.e(e, "Failed to refresh cache")
        }
    }
    
    /**
     * Read market config from cache or local file
     * This is the primary method for other modules to get model data
     */
    suspend fun readMarketConfig(modelId: String): ModelMarketItem? {
        // Ensure cache is initialized
        val context = ApplicationProvider.get()
        if (!isInitialized) {
            initialize(context)
        }
        
        // Try getting from memory cache first
        val cachedItem = marketItemsCache[modelId]
        if (cachedItem != null) {
            return cachedItem
        }
        
        // If not in cache, try reading from local file (fallback)
        return readMarketConfigFromLocal(modelId)
    }
    
    /**
     * Read market config from local disk file (fallback)
     */
    suspend fun readMarketConfigFromLocal(modelId: String): ModelMarketItem? {
        return withContext(Dispatchers.IO) {
            val marketConfigFile = ModelConfig.getMarketConfigFile(modelId)
            Timber.d("readMarketConfigFromLocal for $modelId: looking for file at $marketConfigFile")
            var marketItem: ModelMarketItem? = null
            try {
                val configFile = File(marketConfigFile)
                Timber.d("Config file exists: ${configFile.exists()}")
                if (configFile.exists()) {
                    val configJson = configFile.readText()
                    Timber.d("Config file content: $configJson")
                    marketItem = Gson().fromJson(configJson, ModelMarketItem::class.java)
                    
                    // If successfully read, update memory cache as well
                    if (marketItem != null) {
                        marketItemsCache[modelId] = marketItem
                    }
                }
            } catch (e: Exception) {
                Timber.w(e, "Failed to load market config for ${modelId}")
            }
            marketItem
        }
    }
    
    /**
     * Write market config (for backwards compatibility, but now just updates memory cache)
     * The actual disk write will be handled by the subscription when ModelRepository updates
     */
    @Deprecated(
        "Direct writes are no longer needed. Use ModelRepository.update() to trigger cache updates.",
        ReplaceWith("updateFromRepository()"),
        DeprecationLevel.WARNING
    )
    fun writeMarketConfig(modelItem: ModelMarketItem, marketVersion: String? = null) {
        // Just update memory cache
        marketItemsCache[modelItem.modelId] = modelItem
        
        // Note: Disk write should be triggered by ModelRepository subscription
        // But we still provide this method for backwards compatibility during migration
        writeMarketConfigToLocal(modelItem, marketVersion)
    }
}

