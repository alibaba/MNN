package com.alibaba.mnnllm.android.modelmarket

import android.content.Context
import android.util.Log
import androidx.preference.PreferenceManager
import com.alibaba.mnnllm.android.mainsettings.MainSettings
import com.google.gson.Gson
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.IOException
import java.util.concurrent.TimeUnit

class ModelRepository(private val context: Context) {

    private val httpClient = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .build()

    private val gson = Gson()
    private var cachedModelMarketData: ModelMarketData? = null
    private var isNetworkRequestAttempted = false
    
    companion object {
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
    }

    suspend fun getModelMarketData(): ModelMarketData? = withContext(Dispatchers.IO) {
        // If not allowed to use network, only use assets
        if (!isAllowNetwork(context)) {
            Log.d(TAG, "Debug: Using model_market.json from assets (network disabled by debug switch)")
            val assetsData = loadFromAssets()
            cachedModelMarketData = assetsData
            return@withContext assetsData
        }
        // If we already have cached data and network request was attempted, return cached data
        if (cachedModelMarketData != null && isNetworkRequestAttempted) {
            Log.d(TAG, "Returning cached model market data")
            return@withContext cachedModelMarketData
        }

        try {
            // Try to get data from network first (only once per app lifecycle)
            if (!isNetworkRequestAttempted) {
                Log.d(TAG, "Attempting to fetch model market data from network")
                val networkData = fetchFromNetwork()
                if (networkData != null) {
                    Log.d(TAG, "Successfully fetched data from network")
                    cachedModelMarketData = networkData
                    saveCacheToFile(networkData)
                    isNetworkRequestAttempted = true
                    return@withContext networkData
                }
                isNetworkRequestAttempted = true
            }

            // If network failed, try to load from local cache
            Log.d(TAG, "Network request failed or not attempted, trying local cache")
            val cacheData = loadFromCache()
            if (cacheData != null) {
                Log.d(TAG, "Successfully loaded data from local cache")
                cachedModelMarketData = cacheData
                return@withContext cacheData
            }

            // If cache also failed, fallback to assets
            Log.d(TAG, "Local cache not available, falling back to assets")
            val assetsData = loadFromAssets()
            if (assetsData != null) {
                Log.d(TAG, "Successfully loaded data from assets")
                cachedModelMarketData = assetsData
                return@withContext assetsData
            }

            Log.e(TAG, "All data sources failed")
            null
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get model market data", e)
            null
        }
    }

    private suspend fun fetchFromNetwork(): ModelMarketData? = withContext(Dispatchers.IO) {
        // If not allowed to use network, skip network and use assets
        if (!isAllowNetwork(context)) {
            Log.d(TAG, "Debug: Skip network, use assets (network disabled by debug switch)")
            return@withContext null
        }
        // Debug: Check if network delay is enabled
        if (isNetworkDelayEnabled(context)) {
            Log.d(TAG, "Debug: Simulating network delay (3 seconds)")
            kotlinx.coroutines.delay(3000)
        }
        try {
            val request = Request.Builder()
                .url(NETWORK_URL)
                .build()

            val response = httpClient.newCall(request).execute()
            
            if (response.isSuccessful) {
                val jsonString = response.body?.string()
                if (!jsonString.isNullOrEmpty()) {
                    val data = gson.fromJson(jsonString, ModelMarketData::class.java)
                    Log.d(TAG, "Successfully parsed network data: $jsonString")
                    return@withContext data
                }
            } else {
                Log.w(TAG, "Network request failed with code: ${response.code}")
            }
        } catch (e: Exception) {
            Log.w(TAG, "Network request failed", e)
        }
        return@withContext null
    }

    private suspend fun loadFromCache(): ModelMarketData? = withContext(Dispatchers.IO) {
        // If not allowed to use network, skip cache and use assets
        if (!isAllowNetwork(context)) {
            Log.d(TAG, "Debug: Skip cache, use assets (network disabled by debug switch)")
            return@withContext null
        }
        try {
            val cacheFile = File(context.filesDir, CACHE_FILE_NAME)
            if (!cacheFile.exists()) {
                Log.d(TAG, "Cache file does not exist")
                return@withContext null
            }

            // Check if cache is still valid
            val sharedPrefs = PreferenceManager.getDefaultSharedPreferences(context)
            val cacheTimestamp = sharedPrefs.getLong(CACHE_TIMESTAMP_KEY, 0)
            val currentTime = System.currentTimeMillis()
            
            if (currentTime - cacheTimestamp > CACHE_VALID_DURATION) {
                Log.d(TAG, "Cache is expired, ignoring cache file")
                return@withContext null
            }

            val jsonString = cacheFile.readText()
            val data = gson.fromJson(jsonString, ModelMarketData::class.java)
            Log.d(TAG, "Successfully loaded data from cache")
            return@withContext data
        } catch (e: Exception) {
            Log.w(TAG, "Failed to load from cache", e)
            return@withContext null
        }
    }

    private suspend fun loadFromAssets(): ModelMarketData? = withContext(Dispatchers.IO) {
        try {
            val inputStream = context.assets.open("model_market.json")
            val jsonString = inputStream.bufferedReader().use { it.readText() }
            val data = gson.fromJson(jsonString, ModelMarketData::class.java)
            Log.d(TAG, "Successfully loaded data from assets")
            return@withContext data
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load from assets", e)
            return@withContext null
        }
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
            
            Log.d(TAG, "Successfully saved data to cache")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save cache", e)
        }
    }

    suspend fun getModels(): List<ModelMarketItem> = withContext(Dispatchers.IO) {
        try {
            val data = getModelMarketData()
            val allItems = data?.models ?: emptyList()
            processModels(allItems)
        } catch (e: IOException) {
            emptyList()
        }
    }

    suspend fun getTtsModels(): List<ModelMarketItem> = withContext(Dispatchers.IO) {
        try {
            val data = getModelMarketData()
            val allItems = data?.ttsModels ?: emptyList()
            processModels(allItems)
        } catch (e: IOException) {
            emptyList()
        }
    }

    suspend fun getAsrModels(): List<ModelMarketItem> = withContext(Dispatchers.IO) {
        try {
            val data = getModelMarketData()
            val allItems = data?.asrModels ?: emptyList()
            processModels(allItems)
        } catch (e: IOException) {
            emptyList()
        }
    }

    private suspend fun processModels(models: List<ModelMarketItem>): List<ModelMarketItem> = withContext(Dispatchers.IO) {
        val selectedSource = MainSettings.getDownloadProviderString(context)
        
        Log.d(TAG, "Processing models with selected source: $selectedSource")
        
        return@withContext models.filter { 
            it.sources.containsKey(selectedSource) 
        }.onEach { item ->
            item.currentSource = selectedSource
            item.currentRepoPath = item.sources[selectedSource]!!
            item.modelId = "$selectedSource/${item.sources[selectedSource]!!}"
        }
    }

    /**
     * Determine the model type
     * @param modelId Model ID
     * @return "ASR" if it's an ASR model, "TTS" if it's a TTS model, "LLM" if it's a regular chat model
     */
    suspend fun getModelType(modelId: String): String = withContext(Dispatchers.IO) {
        try {
            val data = getModelMarketData()
            val selectedSource = MainSettings.getDownloadProviderString(context)
            
            data?.let { marketData ->
                marketData.asrModels?.any { model ->
                    model.currentSource = selectedSource
                    model.currentRepoPath = model.sources[selectedSource] ?: ""
                    model.modelId = "$selectedSource/${model.currentRepoPath}"
                    Log.d(TAG, "process ASR model: ${model.modelId}")
                    model.modelId == modelId
                } == true && return@withContext "ASR"
                
                marketData.ttsModels?.any { model ->
                    model.currentSource = selectedSource
                    model.currentRepoPath = model.sources[selectedSource] ?: ""
                    model.modelId = "$selectedSource/${model.currentRepoPath}"
                    Log.d(TAG, "process TTS model: ${model.modelId}")
                    model.modelId == modelId
                } == true && return@withContext "TTS"
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to determine model type for $modelId", e)
        }
        return@withContext "LLM"
    }

} 