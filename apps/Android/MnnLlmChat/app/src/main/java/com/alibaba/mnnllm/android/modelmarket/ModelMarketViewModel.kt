package com.alibaba.mnnllm.android.modelmarket

import android.app.Application
import android.os.Handler
import android.text.TextUtils
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.viewModelScope
import com.alibaba.mls.api.download.DownloadInfo
import com.alibaba.mls.api.download.DownloadListener
import com.alibaba.mls.api.download.DownloadState
import com.alibaba.mls.api.download.ModelDownloadManager
import com.alibaba.mnnllm.android.model.Modality
import com.alibaba.mnnllm.android.modelmarket.ModelMarketCache
import kotlinx.coroutines.launch
import android.content.Intent
import android.os.Build
import com.alibaba.mls.api.download.DownloadForegroundService
import java.util.Locale

class ModelMarketViewModel(application: Application) : AndroidViewModel(application), DownloadListener {

    companion object {
        private const val TAG = "ModelMarketViewModel"
    }

    private val downloadManager = ModelDownloadManager.getInstance(application)
    private var allModels: List<ModelMarketItemWrapper> = emptyList()
    private var mainHandler: Handler = Handler(application.mainLooper)
    private val lastUpdateTimeMap: MutableMap<String, Long> = HashMap()
    private val lastDownloadProgressStage: MutableMap<String, String> = HashMap()
    private var modelMarketConfig: ModelMarketConfig? = null
    private var currentFilterState: FilterState = FilterState()

    private val _models = MutableLiveData<List<ModelMarketItemWrapper>>()
    val models: LiveData<List<ModelMarketItemWrapper>> = _models

    private val _progressUpdate = MutableLiveData<Pair<String, DownloadInfo>>()
    val progressUpdate: LiveData<Pair<String, DownloadInfo>> = _progressUpdate

    private val _itemUpdate = MutableLiveData<String>()
    val itemUpdate: LiveData<String> = _itemUpdate

    // Add loading and error states
    private val _isLoading = MutableLiveData<Boolean>()
    val isLoading: LiveData<Boolean> = _isLoading

    private val _loadError = MutableLiveData<String?>()
    val loadError: LiveData<String?> = _loadError

    init {
        loadModels()
        downloadManager.addListener(this)
    }

    private fun updateDownloadInfo(modelId: String, downloadInfo: DownloadInfo) {
        val currentList = allModels.map { it.copy() }
        currentList.find { it.modelMarketItem.modelId == modelId }?.let {
            it.downloadInfo = downloadInfo
        }
        allModels = currentList
        // Re-apply current filter state instead of showing all models
        applyCurrentFilters()
    }
    
    private fun applyCurrentFilters() {
        var filteredList = allModels

        if (currentFilterState.tagKeys.isNotEmpty()) {
            filteredList = filteredList.filter { wrapper ->
                currentFilterState.tagKeys.all { tagKey -> 
                    wrapper.modelMarketItem.structuredTags.any { it.key == tagKey }
                }
            }
        }

        if (currentFilterState.vendors.isNotEmpty()) {
            filteredList = filteredList.filter { wrapper ->
                currentFilterState.vendors.contains(wrapper.modelMarketItem.vendor)
            }
        }

        if (currentFilterState.size != null) {
            val (min, max) = when (currentFilterState.size) {
                "<1B" -> 0.0 to 1.0
                "1B-5B" -> 1.0 to 5.0
                "5B-15B" -> 5.0 to 15.0
                ">15B" -> 15.0 to Double.MAX_VALUE
                else -> 0.0 to Double.MAX_VALUE
            }
            filteredList = filteredList.filter { it.modelMarketItem.sizeB >= min && it.modelMarketItem.sizeB < max }
        }

        if (currentFilterState.modality != null) {
            filteredList = filteredList.filter { wrapper ->
                Modality.checkModality(wrapper.modelMarketItem.modelId.lowercase(),
                    currentFilterState.modality!!
                )
            }
        }

        if (currentFilterState.downloadState != null) {
            Log.d(TAG, "Filtering for download state: ${currentFilterState.downloadState}. Total models before filter: ${filteredList.size}")
            
            filteredList = filteredList.filter { wrapper ->
                wrapper.downloadInfo.downloadState == currentFilterState.downloadState
            }
            
            Log.d(TAG, "Models after download state filter: ${filteredList.size}")
        }

        if (currentFilterState.source != null) {
            filteredList = filteredList.filter { wrapper ->
                wrapper.modelMarketItem.sources.containsKey(currentFilterState.source)
            }
        }

        // Apply search query filter
        if (currentFilterState.searchQuery.isNotEmpty()) {
            val searchLower = currentFilterState.searchQuery.lowercase(Locale.getDefault())
            filteredList = filteredList.filter { wrapper ->
                val modelMarketItem = wrapper.modelMarketItem
                // Search in model name, description, tags, etc.
                modelMarketItem.modelName.lowercase(Locale.getDefault()).contains(searchLower) ||
                modelMarketItem.description?.lowercase(Locale.getDefault())?.contains(searchLower) == true ||
                modelMarketItem.tags.any { it.lowercase(Locale.getDefault()).contains(searchLower) }
            }
        }

        _models.postValue(filteredList)
    }

    fun loadModels() {
        viewModelScope.launch {
            try {
                // Set loading state
                _isLoading.postValue(true)
                _loadError.postValue(null)
                
                // Get pre-processed config with all models
                val config = ModelRepository.getModelMarketData()
                modelMarketConfig = config
                
                if (config == null) {
                    // Handle case where data couldn't be loaded
                    _isLoading.postValue(false)
                    _loadError.postValue("Failed to load model data")
                    return@launch
                }
                
                // Initialize TagMapper with config
                TagMapper.initializeFromConfig(config)
                
                // Use pre-processed models directly from config
                val marketItems = config.llmModels
                allModels = marketItems.map {
                    val downloadInfo = downloadManager.getDownloadInfo(it.modelId)
                    // If download hasn't started and we have file size in metadata, use it
                    if (downloadInfo.totalSize == 0L && it.fileSize > 0) {
                        downloadInfo.totalSize = it.fileSize
                    }
                    Log.d(TAG, "Model: ${it.modelName}, modelId: ${it.modelId} initial downloadState: ${downloadInfo.downloadState}")
                    ModelMarketItemWrapper(it, downloadInfo)
                }
                Log.d(TAG, "allModels: ${allModels.size} models loaded")
                
                // Clear loading state
                _isLoading.postValue(false)
                applyCurrentFilters()
                
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load models", e)
                _isLoading.postValue(false)
                _loadError.postValue("Loading failed: ${e.message}")
            }
        }
    }



    fun applyFilters(filterState: FilterState) {
        currentFilterState = filterState
        applyCurrentFilters()
    }

    private fun updateServiceState() {
        val downloadingItems = allModels.filter {
            it.downloadInfo.downloadState == DownloadState.DOWNLOADING
        }
        val count = downloadingItems.size
        val context = getApplication<Application>()
        val intent = Intent(context, DownloadForegroundService::class.java)

        if (count > 0) {
            val name = downloadingItems.firstOrNull()?.modelMarketItem?.modelName
            intent.putExtra(DownloadForegroundService.EXTRA_DOWNLOAD_COUNT, count)
            intent.putExtra(DownloadForegroundService.EXTRA_MODEL_NAME, name)

            try {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    context.startForegroundService(intent)
                } else {
                    context.startService(intent)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to start download service", e)
            }
        } else {
            context.stopService(intent)
        }
    }

    fun startDownload(item: ModelMarketItem) {
        downloadManager.startDownload(item.modelId)
    }

    fun pauseDownload(item: ModelMarketItem) {
        downloadManager.pauseDownload(item.modelId)
    }

    fun getDownloadInfo(item: ModelMarketItem): DownloadInfo {
        return downloadManager.getDownloadInfo(item.modelId)
    }

    fun deleteModel(item: ModelMarketItem) {
        viewModelScope.launch {
            downloadManager.deleteModel(item.modelId)
        }
    }

    fun updateModel(item: ModelMarketItem) {
        downloadManager.startDownload(item.modelId)
    }

    fun getAvailableVendors(): List<String> {
        val availableVendors = allModels.map { it.modelMarketItem.vendor }.distinct()
        val vendorOrder = modelMarketConfig?.vendorOrder ?: emptyList()
        
        // Sort vendors according to the custom order, with unlisted vendors at the end
        return availableVendors.sortedWith { a, b ->
            val indexA = vendorOrder.indexOf(a)
            val indexB = vendorOrder.indexOf(b)
            
            when {
                indexA == -1 && indexB == -1 -> a.compareTo(b) // Both not in order list, sort alphabetically
                indexA == -1 -> 1 // a not in order list, put it after b
                indexB == -1 -> -1 // b not in order list, put it after a
                else -> indexA.compareTo(indexB) // Both in order list, sort by index
            }
        }
    }

    fun getAvailableTags(): List<Tag> {
        return allModels.flatMap { it.modelMarketItem.structuredTags }
            .distinctBy { it.key }
            .sortedBy { it.getDisplayText() }
    }
    
    fun getQuickFilterTags(): List<Tag> {
        return modelMarketConfig?.let { config ->
            TagMapper.getQuickFilterTags(config.quickFilterTags)
        } ?: emptyList()
    }

    fun handleItemClick(modelItem: ModelMarketItem, downloadInfo: DownloadInfo) {
        when (downloadInfo.downloadState) {
            DownloadState.COMPLETED -> {
                // This part needs to be implemented to run the model
            }
            DownloadState.NOT_START,
            DownloadState.FAILED, 
            DownloadState.PAUSED -> {
                startDownload(modelItem)
            }
            DownloadState.DOWNLOADING -> {
                // Already downloading, could show a toast or do nothing
            }
        }
    }

    // Utility to write market config JSON after download

    // DownloadListener implementation - reusing logic from ModelListPresenter
    override fun onDownloadTotalSize(modelId: String, totalSize: Long) {
        mainHandler.post {
            allModels.find { it.modelMarketItem.modelId == modelId }?.let { updateDownloadInfo(modelId, downloadManager.getDownloadInfo(it.modelMarketItem.modelId)) }
            _itemUpdate.value = modelId
        }
    }

    override fun onDownloadHasUpdate(modelId: String, downloadInfo: DownloadInfo) {
        mainHandler.post {
            allModels.find { it.modelMarketItem.modelId == modelId }?.let { updateDownloadInfo(modelId, downloadInfo) }
            _itemUpdate.value = modelId
        }
    }

    override fun onDownloadStart(modelId: String) {
        lastDownloadProgressStage.remove(modelId)
        lastUpdateTimeMap.remove(modelId)
        mainHandler.post {
            allModels.find { it.modelMarketItem.modelId == modelId }?.let { updateDownloadInfo(modelId, downloadManager.getDownloadInfo(it.modelMarketItem.modelId)) }
            _itemUpdate.value = modelId
            updateServiceState()
        }
    }

    override fun onDownloadFailed(modelId: String, e: Exception) {
        Log.d(TAG, "[onDownloadFailed] Received for modelId: $modelId, error: ${e.message}")
        mainHandler.post {
            val wrapper = allModels.find { it.modelMarketItem.modelId == modelId }
            if (wrapper != null) {
                Log.d(TAG, "[onDownloadFailed] Found wrapper for $modelId. Updating info.")
                updateDownloadInfo(modelId, downloadManager.getDownloadInfo(wrapper.modelMarketItem.modelId))
                _itemUpdate.value = modelId
                Log.d(TAG, "[onDownloadFailed] Fired _itemUpdate for $modelId.")
            } else {
                Log.e(TAG, "[onDownloadFailed] Could not find wrapper for modelId: $modelId")
            }
            updateServiceState()
        }
    }

    override fun onDownloadProgress(modelId: String, downloadInfo: DownloadInfo) {
        Log.v(TAG, "[onDownloadProgress] Entry point for $modelId. Stage: ${downloadInfo.progressStage}, Progress: ${downloadInfo.progress}")
        val lastUpdateTime = lastUpdateTimeMap[modelId]
        val now = System.currentTimeMillis()
        val progressStateChanged = !TextUtils.equals(
            downloadInfo.progressStage,
            lastDownloadProgressStage[modelId]
        )
        Log.d(TAG, "[onDownloadProgress] Progress state changed: $progressStateChanged lastUpdateTime=$lastUpdateTime, now=$now")
//        if (!progressStateChanged && lastUpdateTime != null && now - lastUpdateTime < 500) {
//            return
//        }
        
        lastDownloadProgressStage[modelId] = downloadInfo.progressStage ?: ""
        lastUpdateTimeMap[modelId] = now

        mainHandler.post {
            Log.d(TAG, "onDownloadProgress: modelId=$modelId, progress=${downloadInfo.progress}")
            // Force a full item update if the progress stage changes (e.g., from nothing to "Preparing")
            // to ensure the UI state (like failed -> downloading) is refreshed correctly.
            if (progressStateChanged) {
                allModels.find { it.modelMarketItem.modelId == modelId }?.let { updateDownloadInfo(modelId, downloadInfo) }
                _itemUpdate.value = modelId
            } else {
                // Just update progress without full refresh
                _progressUpdate.value = Pair(modelId, downloadInfo)
            }
            
            if (progressStateChanged) {
                 updateServiceState()
            }
        }
    }

    override fun onDownloadFinished(modelId: String, path: String) {
        mainHandler.post {
            allModels.find { it.modelMarketItem.modelId == modelId }?.let {
                updateDownloadInfo(modelId, downloadManager.getDownloadInfo(it.modelMarketItem.modelId))
                // Note: Market config write is now handled automatically by ModelMarketCache
                // which subscribes to ModelRepository changes
            }
            _itemUpdate.value = modelId
            updateServiceState()
        }
    }

    override fun onDownloadPaused(modelId: String) {
        mainHandler.post {
            allModels.find { it.modelMarketItem.modelId == modelId }?.let { updateDownloadInfo(modelId, downloadManager.getDownloadInfo(it.modelMarketItem.modelId)) }
            _itemUpdate.value = modelId
            updateServiceState()
        }
    }

    override fun onDownloadFileRemoved(modelId: String) {
        mainHandler.post {
            allModels.find { it.modelMarketItem.modelId == modelId }?.let { updateDownloadInfo(modelId, downloadManager.getDownloadInfo(it.modelMarketItem.modelId)) }
            _itemUpdate.value = modelId
            updateServiceState()
        }
    }

    override fun onCleared() {
        super.onCleared()
        downloadManager.removeListener(this)
        mainHandler.removeCallbacksAndMessages(null)
    }
} 