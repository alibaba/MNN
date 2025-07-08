package com.alibaba.mnnllm.android.modelmarket

import android.content.Context
import android.util.Log
import androidx.preference.PreferenceManager
import com.google.gson.Gson
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.IOException

class ModelRepository(private val context: Context) {

    suspend fun getModelMarketData(): ModelMarketData? = withContext(Dispatchers.IO) {
        try {
            val inputStream = context.assets.open("model_market.json")
            val jsonString = inputStream.bufferedReader().use { it.readText() }
            val gson = Gson()
            val data = gson.fromJson(jsonString, ModelMarketData::class.java)
            data
        } catch (e: Exception) {
            Log.e(TAG, "Failed to parse model_market.json", e)
            null
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
        val sharedPrefs = PreferenceManager.getDefaultSharedPreferences(context)
        val selectedSource = sharedPrefs.getString(
            SourceSelectionDialogFragment.KEY_SOURCE, 
            SourceSelectionDialogFragment.SOURCE_MODELSCOPE
        )!!
        
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
            val selectedSource = getSelectedSource()
            
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

    private fun getSelectedSource(): String {
        val sharedPrefs = PreferenceManager.getDefaultSharedPreferences(context)
        return sharedPrefs.getString(
            SourceSelectionDialogFragment.KEY_SOURCE, 
            SourceSelectionDialogFragment.SOURCE_MODELSCOPE
        )!!
    }

    companion object {
        private const val TAG = "ModelRepository"
    }
} 