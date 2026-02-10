package com.alibaba.mnnllm.android.chat.voice

import android.app.Application
import android.os.Handler
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.viewModelScope
import com.alibaba.mls.api.download.DownloadInfo
import com.alibaba.mls.api.download.DownloadListener
import com.alibaba.mls.api.download.ModelDownloadManager
import com.alibaba.mnnllm.android.modelmarket.ModelMarketItem
import com.alibaba.mnnllm.android.modelmarket.ModelMarketItemWrapper
import com.alibaba.mnnllm.android.modelmarket.ModelRepository
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class VoiceModelMarketViewModel(application: Application) : AndroidViewModel(application), DownloadListener {

    companion object {
        private const val TAG = "VoiceModelMarketViewModel"
    }

    private val downloadManager = ModelDownloadManager.getInstance(application)
    private var allTtsModels: List<ModelMarketItemWrapper> = emptyList()
    private var allAsrModels: List<ModelMarketItemWrapper> = emptyList()
    private var mainHandler: Handler = Handler(application.mainLooper)
    
    private val _models = MutableLiveData<List<ModelMarketItemWrapper>>()
    val models: LiveData<List<ModelMarketItemWrapper>> = _models

    private val _progressUpdate = MutableLiveData<Pair<String, DownloadInfo>>()
    val progressUpdate: LiveData<Pair<String, DownloadInfo>> = _progressUpdate

    private val _itemUpdate = MutableLiveData<String>()
    val itemUpdate: LiveData<String> = _itemUpdate

    init {
        downloadManager.addListener(this)
    }

    fun loadTtsModels() {
        viewModelScope.launch {
            try {
                val ttsItems = ModelRepository.getMarketDataSuspend().ttsModels
                allTtsModels = processModels(ttsItems)
                _models.postValue(allTtsModels)
                Log.d(TAG, "Loaded ${allTtsModels.size} TTS models")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load TTS models", e)
                _models.postValue(emptyList())
            }
        }
    }

    fun loadAsrModels() {
        viewModelScope.launch {
            try {
                val asrItems = ModelRepository.getMarketDataSuspend().asrModels
                allAsrModels = processModels(asrItems)
                _models.postValue(allAsrModels)
                Log.d(TAG, "Loaded ${allAsrModels.size} ASR models")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load ASR models", e)
                _models.postValue(emptyList())
            }
        }
    }

    private suspend fun processModels(models: List<ModelMarketItem>): List<ModelMarketItemWrapper> = withContext(Dispatchers.IO) {
        return@withContext models.map { item ->
            // Get download info
            val downloadInfo = downloadManager.getDownloadInfo(item.modelId)
            Log.d(TAG, "Model: ${item.modelName}, modelId: ${item.modelId}, downloadState: ${downloadInfo.downloadState}")
            
            ModelMarketItemWrapper(item, downloadInfo)
        }
    }

    private fun updateDownloadInfo(modelId: String) {
        // Update TTS models
        allTtsModels.find { it.modelMarketItem.modelId == modelId }?.let { wrapper ->
            wrapper.downloadInfo = downloadManager.getDownloadInfo(wrapper.modelMarketItem.modelId)
        }
        
        // Update ASR models  
        allAsrModels.find { it.modelMarketItem.modelId == modelId }?.let { wrapper ->
            wrapper.downloadInfo = downloadManager.getDownloadInfo(wrapper.modelMarketItem.modelId)
        }
        
        // Update current displayed list
        val currentList = _models.value ?: emptyList()
        val updatedList = currentList.map { wrapper ->
            if (wrapper.modelMarketItem.modelId == modelId) {
                wrapper.downloadInfo = downloadManager.getDownloadInfo(wrapper.modelMarketItem.modelId)
            }
            wrapper
        }
        _models.postValue(updatedList)
    }

    fun startDownload(item: ModelMarketItem) {
        Log.d(TAG, "Starting download for: ${item.modelId}")
        downloadManager.startDownload(item.modelId)
    }

    fun pauseDownload(item: ModelMarketItem) {
        Log.d(TAG, "Pausing download for: ${item.modelId}")
        downloadManager.pauseDownload(item.modelId)
    }

    fun deleteModel(item: ModelMarketItem) {
        viewModelScope.launch {
            Log.d(TAG, "Deleting model: ${item.modelId}")
            downloadManager.deleteModel(item.modelId)
        }
    }

    fun updateModel(item: ModelMarketItem) {
        Log.d(TAG, "Starting update for: ${item.modelId}")
        downloadManager.startDownload(item.modelId)
    }

    // DownloadListener implementation
    override fun onDownloadTotalSize(modelId: String, totalSize: Long) {
        mainHandler.post {
            updateDownloadInfo(modelId)
            _itemUpdate.value = modelId
        }
    }

    override fun onDownloadHasUpdate(modelId: String, downloadInfo: DownloadInfo) {
        mainHandler.post {
            updateDownloadInfo(modelId)
            _itemUpdate.value = modelId
        }
    }


    override fun onDownloadStart(modelId: String) {
        Log.d(TAG, "Download started for: $modelId")
        mainHandler.post {
            updateDownloadInfo(modelId)
            _itemUpdate.value = modelId
        }
    }

    override fun onDownloadProgress(modelId: String, downloadInfo: DownloadInfo) {
        mainHandler.post {
            _progressUpdate.value = Pair(modelId, downloadInfo)
        }
    }

    override fun onDownloadFailed(modelId: String, exception: Exception) {
        Log.d(TAG, "Download failed for: $modelId, error: ${exception.message}")
        mainHandler.post {
            updateDownloadInfo(modelId)
            _itemUpdate.value = modelId
        }
    }

    override fun onDownloadFinished(modelId: String, path: String) {
        Log.d(TAG, "Download finished for: $modelId")
        mainHandler.post {
            updateDownloadInfo(modelId)
            _itemUpdate.value = modelId
        }
    }

    override fun onDownloadPaused(modelId: String) {
        Log.d(TAG, "Download paused for: $modelId")
        mainHandler.post {
            updateDownloadInfo(modelId)
            _itemUpdate.value = modelId
        }
    }

    override fun onDownloadFileRemoved(modelId: String) {
        Log.d(TAG, "Download file removed for: $modelId")
        mainHandler.post {
            updateDownloadInfo(modelId)
            _itemUpdate.value = modelId
        }
    }

    fun setDefaultTtsModel(modelId: String) {
        val context = getApplication<Application>()
        com.alibaba.mnnllm.android.mainsettings.MainSettings.setDefaultTtsModel(context, modelId)
        Log.d(TAG, "Set default TTS model: $modelId")
    }

    fun setDefaultAsrModel(modelId: String) {
        val context = getApplication<Application>()
        com.alibaba.mnnllm.android.mainsettings.MainSettings.setDefaultAsrModel(context, modelId)
        Log.d(TAG, "Set default ASR model: $modelId")
    }

    override fun onCleared() {
        super.onCleared()
        downloadManager.removeListener(this)
        mainHandler.removeCallbacksAndMessages(null)
    }
} 