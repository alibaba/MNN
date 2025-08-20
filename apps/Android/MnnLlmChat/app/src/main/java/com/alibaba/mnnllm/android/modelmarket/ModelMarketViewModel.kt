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
import com.alibaba.mnnllm.android.modelmarket.ModelMarketUtils.writeMarketConfig
import kotlinx.coroutines.launch
import java.util.Locale

class ModelMarketViewModel(application: Application) : AndroidViewModel(application), DownloadListener {

    companion object {
        private const val TAG = "ModelMarketViewModel"
    }

    private val repository = ModelRepository(application)
    private val downloadManager = ModelDownloadManager.getInstance(application)
    private var allModels: List<ModelMarketItemWrapper> = emptyList()
    private var mainHandler: Handler = Handler(application.mainLooper)
    private val lastUpdateTimeMap: MutableMap<String, Long> = HashMap()
    private val lastDownloadProgressStage: MutableMap<String, String> = HashMap()
    private var modelMarketData: ModelMarketData? = null
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

        if (currentFilterState.downloadState == "true") {
            Log.d(TAG, "Filtering for completed downloads. Total models before filter: ${filteredList.size}")
            
            // Debug: Print download states of all models
            filteredList.forEach { wrapper ->
                Log.d(TAG, "Model: ${wrapper.modelMarketItem.modelName}, downloadState: ${wrapper.downloadInfo.downloadState}, COMPLETED: ${DownloadState.COMPLETED}")
            }
            
            filteredList = filteredList.filter { wrapper ->
                wrapper.downloadInfo.downloadState == DownloadState.COMPLETED
            }
            
            Log.d(TAG, "Models after completed filter: ${filteredList.size}")
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
                
                // Load full data for TagMapper initialization
                modelMarketData = repository.getModelMarketData()
                
                if (modelMarketData == null) {
                    // Handle case where data couldn't be loaded
                    _isLoading.postValue(false)
                    _loadError.postValue("Failed to load model data")
                    return@launch
                }
                
                modelMarketData?.let { data ->
                    // Initialize TagMapper with data from JSON
                    TagMapper.initializeFromData(data)
                }
                
                // Get filtered models with correctly set modelId
                val marketItems = repository.getModels()
                allModels = marketItems.map {
                    val downloadInfo = downloadManager.getDownloadInfo(it)
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

    fun startDownload(item: ModelMarketItem) {
        downloadManager.startDownload(item)
    }

    fun pauseDownload(item: ModelMarketItem) {
        downloadManager.pauseDownload(item.modelId)
    }

    fun getDownloadInfo(item: ModelMarketItem): DownloadInfo {
        return downloadManager.getDownloadInfo(item)
    }

    fun deleteModel(item: ModelMarketItem) {
        viewModelScope.launch {
            downloadManager.deleteModel(item)
        }
    }

    fun updateModel(item: ModelMarketItem) {
        downloadManager.startDownload(item)
    }

    fun getAvailableVendors(): List<String> {
        val availableVendors = allModels.map { it.modelMarketItem.vendor }.distinct()
        val vendorOrder = modelMarketData?.vendorOrder ?: emptyList()
        
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
        return modelMarketData?.let { data ->
            TagMapper.getQuickFilterTags(data.quickFilterTags)
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
            allModels.find { it.modelMarketItem.modelId == modelId }?.let { updateDownloadInfo(modelId, downloadManager.getDownloadInfo(it.modelMarketItem)) }
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
            allModels.find { it.modelMarketItem.modelId == modelId }?.let { updateDownloadInfo(modelId, downloadManager.getDownloadInfo(it.modelMarketItem)) }
            _itemUpdate.value = modelId
        }
    }

    override fun onDownloadFailed(modelId: String, e: Exception) {
        Log.d(TAG, "[onDownloadFailed] Received for modelId: $modelId, error: ${e.message}")
        mainHandler.post {
            val wrapper = allModels.find { it.modelMarketItem.modelId == modelId }
            if (wrapper != null) {
                Log.d(TAG, "[onDownloadFailed] Found wrapper for $modelId. Updating info.")
                updateDownloadInfo(modelId, downloadManager.getDownloadInfo(wrapper.modelMarketItem))
                _itemUpdate.value = modelId
                Log.d(TAG, "[onDownloadFailed] Fired _itemUpdate for $modelId.")
            } else {
                Log.e(TAG, "[onDownloadFailed] Could not find wrapper for modelId: $modelId")
            }
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
        }
    }

    override fun onDownloadFinished(modelId: String, path: String) {
        mainHandler.post {
            allModels.find { it.modelMarketItem.modelId == modelId }?.let {
                updateDownloadInfo(modelId, downloadManager.getDownloadInfo(it.modelMarketItem))
                // Write market config after download
                writeMarketConfig(it.modelMarketItem)
            }
            _itemUpdate.value = modelId
        }
    }

    override fun onDownloadPaused(modelId: String) {
        mainHandler.post {
            allModels.find { it.modelMarketItem.modelId == modelId }?.let { updateDownloadInfo(modelId, downloadManager.getDownloadInfo(it.modelMarketItem)) }
            _itemUpdate.value = modelId
        }
    }

    override fun onDownloadFileRemoved(modelId: String) {
        mainHandler.post {
            allModels.find { it.modelMarketItem.modelId == modelId }?.let { updateDownloadInfo(modelId, downloadManager.getDownloadInfo(it.modelMarketItem)) }
            _itemUpdate.value = modelId
        }
    }

    override fun onCleared() {
        super.onCleared()
        downloadManager.removeListener(this)
        mainHandler.removeCallbacksAndMessages(null)
    }
} 