package com.alibaba.mnnllm.android.modelmarket

import android.content.Context
import android.util.Log
import androidx.compose.runtime.rememberUpdatedState
import androidx.preference.PreferenceManager
import com.alibaba.mls.api.download.DownloadPersistentData
import com.alibaba.mnnllm.android.mainsettings.MainSettings
import com.google.gson.Gson
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.TimeUnit

object ModelRepository {

    private val context: Context by lazy { 
        com.alibaba.mnnllm.android.MnnLlmApplication.getAppContext() 
    }
    
    private val httpClient = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .cache(null) // Disable HTTP cache
        .build()

    private val gson = Gson()
    private var cachedModelMarketData: ModelMarketConfig? = null
    private var isNetworkRequestAttempted = false
    
    // Flow state management
    private val _marketDataFlow = MutableStateFlow<ModelMarketConfig?>(null)
    private val _loadingStateFlow = MutableStateFlow(LoadingState.IDLE)
    private val _errorFlow = MutableStateFlow<Throwable?>(null)
    
    val marketDataFlow: StateFlow<ModelMarketConfig?> = _marketDataFlow.asStateFlow()
    val loadingStateFlow: StateFlow<LoadingState> = _loadingStateFlow.asStateFlow()
    val errorFlow: StateFlow<Throwable?> = _errorFlow.asStateFlow()
    
    // ConcurrentHashMap cache for fast lookups
    private val marketItemsCache = ConcurrentHashMap<String, ModelMarketItem>()
    private val initializationMutex = Mutex()
    
    // Background refresh management
    private var refreshJob: Job? = null
    private val refreshScope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    
    enum class LoadingState {
        IDLE,           // Not loading
        LOADING_CACHE,  // Loading from cache/assets
        LOADING_NETWORK, // Loading from network
        LOADED,         // Successfully loaded
        ERROR           // Error occurred
    }
    
    
    private const val TAG = "ModelRepository"
    private const val NETWORK_URL = "https://meta.alicdn.com/data/mnn/apis/model_market.json"
    private const val CACHE_FILE_NAME = "model_market_cache.json"
    private const val CACHE_TIMESTAMP_KEY = "model_market_cache_timestamp"
    private const val CACHE_VALID_DURATION = 1 * 60 * 60 * 1000L 
    private const val KEY_ALLOW_NETWORK_MARKET_DATA = "debug_allow_network_market_data"
    
    private fun isAllowNetwork(context: Context): Boolean {
        return com.alibaba.mnnllm.android.utils.PreferenceUtils.getBoolean(context, KEY_ALLOW_NETWORK_MARKET_DATA, true)
    }
    
    private fun isNetworkDelayEnabled(context: Context): Boolean {
        return com.alibaba.mnnllm.android.utils.PreferenceUtils.getBoolean(context, "debug_enable_network_delay", false)
    }

    fun getModelMarketDataV2(): ModelMarketConfig? {
        return marketDataFlow.value
    }

    suspend fun getModelMarketData(): ModelMarketConfig? = withContext(Dispatchers.IO) {
        // If not allowed to use network, only use assets
        if (!isAllowNetwork(context)) {
            Log.d(TAG, "Debug: Using model_market.json from assets (network disabled by debug switch)")
            val assetsData = loadFromAssets()
            if (assetsData != null) {
                val config = convertToConfig(assetsData)
                cachedModelMarketData = config
                return@withContext config
            }
            return@withContext null
        }
        // If we already have cached data and network request was attempted, return cached data
        if (cachedModelMarketData != null && isNetworkRequestAttempted) {
            return@withContext cachedModelMarketData
        }

        try {
            // Load assets data first for version comparison
            val assetsData = loadFromAssets()
            
            // Try to get data from network first (only once per app lifecycle)
            if (!isNetworkRequestAttempted) {
                val networkData = fetchFromNetwork()
                if (networkData != null) {
                    // Compare versions
                    if (assetsData != null) {
                        val networkVersion = networkData.version ?: "0"
                        val assetsVersion = assetsData.version
                        
                        if (isVersionLower(networkVersion, assetsVersion)) {
                            val config = convertToConfig(assetsData)
                            cachedModelMarketData = config
                            isNetworkRequestAttempted = true
                            return@withContext config
                        }
                    }
                    
                    val config = convertToConfig(networkData)
                    cachedModelMarketData = config
                    saveCacheToFile(networkData)
                    isNetworkRequestAttempted = true
                    return@withContext config
                }
                isNetworkRequestAttempted = true
            }

            // If network failed, try to load from local cache
            val cacheData = loadFromCache()
            if (cacheData != null) {
                if (assetsData != null) {
                    val cacheVersion = cacheData.version
                    val assetsVersion = assetsData.version
                    if (isVersionLower(cacheVersion, assetsVersion)) {
                        val config = convertToConfig(assetsData)
                        cachedModelMarketData = config
                        return@withContext config
                    }
                }
                val config = convertToConfig(cacheData)
                cachedModelMarketData = config
                return@withContext config
            }

            // If cache also failed, fallback to assets
            if (assetsData != null) {
                val config = convertToConfig(assetsData)
                cachedModelMarketData = config
                return@withContext config
            }

            null
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get model market data", e)
            null
        }
    }

    private suspend fun fetchFromNetwork(): ModelMarketData? = withContext(Dispatchers.IO) {
        // If not allowed to use network, skip network and use assets
        if (!isAllowNetwork(context)) {
            return@withContext null
        }
        // Debug: Check if network delay is enabled
        if (isNetworkDelayEnabled(context)) {
            kotlinx.coroutines.delay(3000)
        }
        try {
            val request = Request.Builder()
                .url(NETWORK_URL)
                .addHeader("Cache-Control", "no-cache, no-store, must-revalidate")
                .addHeader("Pragma", "no-cache")
                .addHeader("Expires", "0")
                .build()

            val response = httpClient.newCall(request).execute()
            
            if (response.isSuccessful) {
                val jsonString = response.body?.string()
                if (!jsonString.isNullOrEmpty()) {
                    val data = gson.fromJson(jsonString, ModelMarketData::class.java)
                    return@withContext data
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Network request failed", e)
        }
        return@withContext null
    }

    /**
     * Force refresh market data from network and update cache.
     * Will still prefer assets if network version is older than assets.
     */
    suspend fun refreshFromNetwork(): ModelMarketConfig? = withContext(Dispatchers.IO) {
        try {
            val assetsData = loadFromAssets()
            val networkData = fetchFromNetwork()
            if (networkData != null) {
                if (assetsData != null) {
                    val networkVersion = networkData.version ?: "0"
                    val assetsVersion = assetsData.version
                    if (isVersionLower(networkVersion, assetsVersion)) {
                        val config = convertToConfig(assetsData)
                        cachedModelMarketData = config
                        isNetworkRequestAttempted = true
                        return@withContext config
                    }
                }
                val config = convertToConfig(networkData)
                cachedModelMarketData = config
                saveCacheToFile(networkData)
                isNetworkRequestAttempted = true
                return@withContext config
            }
        } catch (e: Exception) {
            Log.w(TAG, "refreshFromNetwork failed", e)
        }
        return@withContext null
    }

    private suspend fun loadFromCache(): ModelMarketData? = withContext(Dispatchers.IO) {
        // If not allowed to use network, skip cache and use assets
        if (!isAllowNetwork(context)) {
            return@withContext null
        }
        
        try {
            val cacheFile = File(context.filesDir, CACHE_FILE_NAME)
            if (!cacheFile.exists()) {
                return@withContext null
            }

            // Check if cache is still valid
            val sharedPrefs = PreferenceManager.getDefaultSharedPreferences(context)
            val cacheTimestamp = sharedPrefs.getLong(CACHE_TIMESTAMP_KEY, 0)
            val currentTime = System.currentTimeMillis()
            val age = currentTime - cacheTimestamp
            
            if (age > CACHE_VALID_DURATION) {
                return@withContext null
            }

            val jsonString = cacheFile.readText()
            val data = gson.fromJson(jsonString, ModelMarketData::class.java)
            return@withContext data
        } catch (e: Exception) {
            Log.w(TAG, "Failed to load from cache: ${e.message}", e)
            return@withContext null
        }
    }

    suspend fun loadFromAssets(): ModelMarketData? = withContext(Dispatchers.IO) {
        try {
            val inputStream = context.assets.open("model_market.json")
            val jsonString = inputStream.bufferedReader().use { it.readText() }
            val data = gson.fromJson(jsonString, ModelMarketData::class.java)
            return@withContext data
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load from assets: ${e.message}", e)
            return@withContext null
        }
    }

    /**
     * Load data from local cache or assets (no network)
     * Compares versions from all sources and returns the highest version
     * When versions are equal, prefer: cache > assets
     */
    suspend fun loadCachedOrAssets(): ModelMarketConfig? = withContext(Dispatchers.IO) {
        val cacheData = loadFromCache()
        val assetsData = loadFromAssets()
        
        // If only one source available, return it after conversion
        if (cacheData == null && assetsData != null) {
            return@withContext convertToConfig(assetsData)
        }
        if (assetsData == null && cacheData != null) {
            return@withContext convertToConfig(cacheData)
        }
        if (cacheData == null && assetsData == null) {
            Log.e(TAG, "CRITICAL: Both cache and assets are NULL! No data available!")
            return@withContext null
        }
        
        // Both available, compare versions
        val cacheVersion = cacheData!!.version
        val assetsVersion = assetsData!!.version
        
        val selectedData = when {
            isVersionLower(cacheVersion, assetsVersion) -> assetsData
            isVersionLower(assetsVersion, cacheVersion) -> cacheData
            else -> cacheData
        }
        
        return@withContext convertToConfig(selectedData)
    }

    private suspend fun saveCacheToFile(data: ModelMarketData) = withContext(Dispatchers.IO) {
        try {
            val cacheFile = File(context.filesDir, CACHE_FILE_NAME)
            val jsonString = gson.toJson(data)
            cacheFile.writeText(jsonString)
            
            // Save timestamp
            val sharedPrefs = PreferenceManager.getDefaultSharedPreferences(context)
            sharedPrefs.edit()
                .putLong(CACHE_TIMESTAMP_KEY, System.currentTimeMillis())
                .apply()
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save cache", e)
        }
    }


    private suspend fun processModels(models: List<ModelMarketItem>): List<ModelMarketItem> = withContext(Dispatchers.IO) {
        val selectedSource = MainSettings.getDownloadProviderString(context)
        
        
        return@withContext models.filter { 
            it.sources.containsKey(selectedSource) 
        }.onEach { item ->
            item.currentSource = selectedSource
            item.currentRepoPath = item.sources[selectedSource]!!
            item.modelId = "$selectedSource/${item.sources[selectedSource]!!}"
            
            // Save file_size to DownloadPersistentData if available
            if (item.fileSize > 0) {
                try {
                    DownloadPersistentData.saveMarketSizeTotalSuspend(context, item.modelId, item.fileSize)
                } catch (e: Exception) {
                    // Failed to save market size
                }
            }
        }
    }

    /**
     * Convert ModelMarketData to ModelMarketConfig with processed models
     */
    private suspend fun convertToConfig(data: ModelMarketData): ModelMarketConfig = withContext(Dispatchers.IO) {
        val llmModels = processModels(data.models)
        val ttsModels = processModels(data.ttsModels ?: emptyList())
        val asrModels = processModels(data.asrModels ?: emptyList())
        val libs = processModels(data.libs ?: emptyList())
        
        ModelMarketConfig(
            version = data.version,
            tagTranslations = data.tagTranslations,
            quickFilterTags = data.quickFilterTags,
            vendorOrder = data.vendorOrder ?: emptyList(),
            llmModels = llmModels,
            ttsModels = ttsModels,
            asrModels = asrModels,
            libs = libs
        )
    }


    /**
     * Determine the model type
     * @param modelId Model ID
     * @return "ASR" if it's an ASR model, "TTS" if it's a TTS model, "LLM" if it's a regular chat model
     */
    suspend fun getModelType(modelId: String): String = withContext(Dispatchers.IO) {
        try {
            val config = getModelMarketData()
            
            config?.let { marketConfig ->
                // Check if it's in ASR models
                if (marketConfig.asrModels.any { it.modelId == modelId }) {
                    return@withContext "ASR"
                }
                
                // Check if it's in TTS models
                if (marketConfig.ttsModels.any { it.modelId == modelId }) {
                    return@withContext "TTS"
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to determine model type for $modelId", e)
        }
        return@withContext "LLM"
    }

    suspend fun getModelTypeSuspend(modelId: String): String {
        return getMarketDataSuspend().let { config ->
            when {
                config.asrModels.find { it.modelId == modelId } != null -> "ASR"
                config.ttsModels.find { it.modelId == modelId } != null -> "TTS"
                config.libs.find { it.modelId == modelId } != null -> "LIB"
                else -> "LLM"
            }
        }
    }

    /**
     * Compare two version strings
     * @param version1 First version string
     * @param version2 Second version string
     * @return true if version1 is lower than version2
     */
    private fun isVersionLower(version1: String, version2: String): Boolean {
        return try {
            val v1 = version1.toInt()
            val v2 = version2.toInt()
            v1 < v2
        } catch (e: NumberFormatException) {
            // If parsing fails, treat as equal (use network data)
            false
        }
    }
    
    /**
     * Initialize the repository with immediate cache access and background network loading
     * This method returns immediately with cached data, then updates with network data in background
     */
    suspend fun initialize(): ModelMarketConfig? = initializationMutex.withLock {
        // Check if already initialized or initializing using LoadingState
        val currentState = _loadingStateFlow.value
        if (currentState != LoadingState.IDLE) {
            return@withLock _marketDataFlow.value
        }
        
        try {
            // Step 1: Load from cache/assets immediately (fast)
            _loadingStateFlow.value = LoadingState.LOADING_CACHE
            val cachedConfig = loadCachedOrAssets()
            
            if (cachedConfig != null) {
                _marketDataFlow.value = cachedConfig
                cachedModelMarketData = cachedConfig
                _loadingStateFlow.value = LoadingState.LOADED
                // Build cache for immediate access
                buildCacheFromData(cachedConfig)
            }
            // Step 2: Start background network refresh (non-blocking)
            startBackgroundRefresh()
            
            return@withLock cachedConfig
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize repository", e)
            _errorFlow.value = e
            _loadingStateFlow.value = LoadingState.ERROR
            return@withLock null
        }
    }
    
    /**
     * Start background network refresh
     */
    private fun startBackgroundRefresh() {
        refreshJob?.cancel()
        refreshJob = refreshScope.launch {
            try {
                _loadingStateFlow.value = LoadingState.LOADING_NETWORK
                
                val networkConfig = refreshFromNetwork()
                
                if (networkConfig != null) {
                    _marketDataFlow.value = networkConfig
                    cachedModelMarketData = networkConfig
                    _loadingStateFlow.value = LoadingState.LOADED
                    _errorFlow.value = null
                    
                    // Update cache
                    buildCacheFromData(networkConfig)
                } else {
                    // Only set ERROR if we don't have any data yet
                    if (_marketDataFlow.value == null) {
                        _loadingStateFlow.value = LoadingState.ERROR
                    }
                }
                
            } catch (e: Exception) {
                Log.e(TAG, "Background network refresh error: ${e.message}", e)
                _errorFlow.value = e
                if (_marketDataFlow.value == null) {
                    _loadingStateFlow.value = LoadingState.ERROR
                }
            }
        }
    }
    
    /**
     * Build ConcurrentHashMap cache from market config for fast lookups
     */
    private suspend fun buildCacheFromData(config: ModelMarketConfig) = withContext(Dispatchers.IO) {
        try {
            // Cache all models directly from config
            marketItemsCache.clear()
            
            // Add regular models
            config.llmModels.forEach { marketItemsCache[it.modelId] = it }
            
            // Add TTS models
            config.ttsModels.forEach { marketItemsCache[it.modelId] = it }
            
            // Add ASR models
            config.asrModels.forEach { marketItemsCache[it.modelId] = it }
            
            // Add libs
            config.libs.forEach { marketItemsCache[it.modelId] = it }
            
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to build cache", e)
        }
    }
    /**
     * Get all cached models (immediate access, no IO)
     */
    fun getAllCachedModels(): List<ModelMarketItem> {
        return marketItemsCache.values.toList()
    }
    
    /**
     * Check if repository is initialized
     */
    fun isInitialized(): Boolean {
        return _loadingStateFlow.value != LoadingState.IDLE
    }
    
    /**
     * Force refresh from network
     */
    suspend fun forceRefresh() {
        if (_loadingStateFlow.value == LoadingState.IDLE) {
            initialize()
            return
        }
        
        startBackgroundRefresh()
    }
    
    /**
     * Subscribe to market data updates
     * Automatically initializes the repository if not already initialized
     * Returns a Flow that emits market data updates
     * 
     * IMPORTANT: This is a suspend function that waits for initialization to complete
     * before returning the flow to avoid race conditions with .first()
     */
    suspend fun subscribeToMarketData(): Flow<ModelMarketConfig> {
        refreshScope.launch {
            initialize()
        }
        // Return a flow that filters out null values
        return marketDataFlow.filterNotNull()
    }

    suspend fun getMarketDataSuspend(): ModelMarketConfig {
        return subscribeToMarketData().first()
    }

    /**
     * Subscribe to loading state changes
     */
    fun subscribeToLoadingState(): Flow<LoadingState> {
        return loadingStateFlow
    }
    
    /**
     * Subscribe to error updates
     */
    fun subscribeToErrors(): Flow<Throwable?> {
        return errorFlow
    }
    
    /**
     * Get current loading state
     */
    fun getCurrentLoadingState(): LoadingState {
        return _loadingStateFlow.value
    }
    
    /**
     * Get tag translations from market data
     * Returns translated tags if available, otherwise returns original tags
     */
    fun getTagTranslations(tags: List<String>): List<String> {
        val translations = _marketDataFlow.value?.tagTranslations ?: emptyMap()
        return tags.map { tag -> translations[tag] ?: tag }
    }
    
    /**
     * Clear all data and reset state
     */
    fun clear() {
        refreshJob?.cancel()
        marketItemsCache.clear()
        _marketDataFlow.value = null
        _loadingStateFlow.value = LoadingState.IDLE  // Reset to IDLE
        _errorFlow.value = null
        cachedModelMarketData = null
        isNetworkRequestAttempted = false
    }
    
    /**
     * Cleanup resources
     */
    fun cleanup() {
        refreshScope.cancel()
        clear()
    }
} 
