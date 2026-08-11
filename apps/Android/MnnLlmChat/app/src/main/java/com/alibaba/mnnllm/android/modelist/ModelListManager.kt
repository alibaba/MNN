package com.alibaba.mnnllm.android.modelist

import android.annotation.SuppressLint
import android.content.Context
import android.os.Build
import com.alibaba.mls.api.ModelItem
import com.alibaba.mls.api.download.DownloadPersistentData
import com.alibaba.mnnllm.android.chat.model.ChatDataManager
import com.alibaba.mnnllm.android.model.ModelTypeUtils
import com.alibaba.mnnllm.android.model.ModelUtils
import com.alibaba.mnnllm.android.MnnLlmApplication
import com.alibaba.mnnllm.android.modelmarket.ModelRepository
import com.alibaba.mnnllm.android.modelmarket.ModelMarketItem
import com.alibaba.mnnllm.android.modelmarket.ModelMarketCache
import com.alibaba.mnnllm.android.modelmarket.TagMapper
import com.alibaba.mnnllm.android.utils.FileUtils
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.utils.PreferenceUtils
import com.google.gson.Gson
import com.google.gson.annotations.SerializedName
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.filterIsInstance
import kotlinx.coroutines.flow.filterNotNull
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import timber.log.Timber
import java.io.File
import java.nio.file.Files
import java.util.Locale
import java.util.concurrent.TimeUnit

@SuppressLint("StaticFieldLeak")
object ModelListManager {
    // Application scope for automatic initialization
    private val applicationScope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    private val marketSyncScope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    private const val TAG = "ModelListManager"
    private const val CACHE_FILE_NAME = "model_list_cache.json"
    private const val CACHE_VERSION = 2
    private const val MAX_CACHE_AGE_DAYS = 9999L
    private const val MODEL_TYPE_LLM = "LLM"
    private const val MODEL_TYPE_UNKNOWN = "UNKNOWN"
    private const val MODEL_TYPE_DIFFUSION = "DIFFUSION"

    // Gson instance for serialization
    private val gson = Gson()

    // Map to store modelId to ModelItem mapping for external access
    private val modelIdModelMap = mutableMapOf<String, ModelItem>()

    // Reentrancy protection mechanism
    private val loadingMutex = Mutex()
    private var cachedModels: List<ModelItemWrapper>? = null

    // Flow-based state management
    private val _modelListState = MutableStateFlow<ModelListState>(ModelListState.Loading)
    val modelListState: StateFlow<ModelListState> = _modelListState.asStateFlow()

    // SharedFlow for one-time events
    private val _refreshEvents = MutableSharedFlow<RefreshEvent>(replay = 0)
    val refreshEvents: SharedFlow<RefreshEvent> = _refreshEvents.asSharedFlow()

    // Cache file and initialization state
    // Cache file location: context.cacheDir/model_list_cache.json
    // This is the Android application's cache directory (typically /data/data/[package_name]/cache/)
    private lateinit var cacheFile: File
    private var hasEmittedInitialCache = false
    private var isInitialized = false
    private var isInitializing = false
    private lateinit var context: Context
    private var marketDataSyncJob: Job? = null
    private var tagWarmupJob: Job? = null
    private var lastSyncedMarketDataKey: String? = null

    // Data classes for Flow state management
    sealed class ModelListState {
        object Loading : ModelListState()
        data class Success(
            val models: List<ModelItemWrapper>,
            val source: DataSource,
            val timestamp: Long = System.currentTimeMillis()
        ) : ModelListState()

        data class Error(val exception: Throwable) : ModelListState()
    }

    enum class DataSource {
        CACHE,      // From disk cache
        FRESH,      // From fresh scan/network
        MEMORY      // From memory cache
    }

    enum class ChangeReason {
        MODEL_DOWNLOADED,
        MODEL_DELETED,
        MODEL_PINNED,
        MODEL_UNPINNED,
        MANUAL_REFRESH,
        MARKET_DATA_UPDATED,
        INITIALIZATION
    }

    sealed class RefreshEvent {
        object Success : RefreshEvent()
        object NoChange : RefreshEvent()
        data class Failed(val exception: Throwable) : RefreshEvent()
    }

    // DTO for disk caching
    data class ModelItemCacheDTO(
        @SerializedName("modelId") val modelId: String,
        @SerializedName("modelPath") val modelPath: String?,
        @SerializedName("downloadSize") val downloadSize: Long,
        @SerializedName("isPinned") val isPinned: Boolean,
        @SerializedName("lastChatTime") val lastChatTime: Long,
        @SerializedName("downloadTime") val downloadTime: Long,
        @SerializedName("isLocal") val isLocal: Boolean,
        @SerializedName("modelName") val modelName: String?,
        @SerializedName("tags") val tags: List<String>?,
        @SerializedName("modelMarketItem") val modelMarketItem: ModelMarketItem? // Cache the complete market item
    ) {
        companion object {
            fun from(wrapper: ModelItemWrapper): ModelItemCacheDTO? {
                val modelId = wrapper.modelItem.modelId
                val modelPath = wrapper.modelItem.localPath

                // CRITICAL: Ensure modelId is never null before saving
                if (modelId == null) {
                    Timber.e("ModelItemCacheDTO.from: modelId is null! Cannot save to cache. ModelItem: ${wrapper.modelItem}")
                    return null
                }

                // Additional validation: ensure modelId is not empty
                if (modelId.isBlank()) {
                    Timber.e("ModelItemCacheDTO.from: modelId is blank! Cannot save to cache. ModelItem: ${wrapper.modelItem}")
                    return null
                }

                // Log models without modelMarketItem for monitoring
                if (wrapper.modelItem.modelMarketItem == null) {
                    Timber.d("ModelItemCacheDTO.from: Model $modelId has no modelMarketItem (will rely on fallback name extraction)")
                }

                // REMOVED: This filtering was too strict and excluded valid models
                // The original logic filtered out non-local models without modelPath,
                // but this is not necessary for caching purposes
                // if (!wrapper.isLocal && modelPath == null) {
                //     Timber.w("from: modelPath is null for non-local model $modelId")
                //     return null
                // }

                Timber.d("ModelItemCacheDTO.from: INCLUDED - modelId: $modelId, isLocal: ${wrapper.isLocal}, modelPath: $modelPath, hasMarketItem: ${wrapper.modelItem.modelMarketItem != null}")

                return ModelItemCacheDTO(
                    modelId = modelId,
                    modelPath = modelPath,
                    downloadSize = wrapper.downloadSize,
                    isPinned = wrapper.isPinned,
                    lastChatTime = wrapper.lastChatTime,
                    downloadTime = wrapper.downloadTime,
                    isLocal = wrapper.isLocal,
                    modelName = (wrapper.modelItem.modelMarketItem as? ModelMarketItem)?.modelName,
                    tags = wrapper.modelItem.getTags(),
                    modelMarketItem = wrapper.modelItem.modelMarketItem as? ModelMarketItem // Cache the complete market item
                )
            }
        }

        fun toWrapper(context: Context): ModelItemWrapper? {
            val modelItem = if (isLocal) {
                val localItem = LocalModelsProvider.getLocalModels().find { it.modelId == modelId }
                if (localItem != null) {
                    if (!tags.isNullOrEmpty()) {
                        localItem.tags = tags
                    }
                    if (localItem.modelMarketItem == null && this.modelMarketItem != null) {
                         localItem.modelMarketItem = this.modelMarketItem
                    }
                    if (!this.modelName.isNullOrEmpty()) {
                        localItem.modelName = this.modelName
                    }
                }
                localItem
            } else {
                try {
                    val item = ModelItem().apply {
                        this.modelId = this@ModelItemCacheDTO.modelId
                        this.localPath = this@ModelItemCacheDTO.modelPath
                        this.modelMarketItem =
                            this@ModelItemCacheDTO.modelMarketItem // Use cached market item
                        this.tags = this@ModelItemCacheDTO.tags ?: emptyList() 
                    }
                    Timber.d("toWrapper: Successfully reconstructed ModelItem for $modelId from cache (modelPath: $modelPath, hasMarketItem: ${item.modelMarketItem != null}, cachedModelId: ${this@ModelItemCacheDTO.modelId})")
                    item
                } catch (e: Exception) {
                    Timber.w(
                        e,
                        "toWrapper: Failed to reconstruct ModelItem for $modelId from cache"
                    )
                    return null
                }
            }

            return modelItem?.let {
                // Validate that modelId is properly set
                if (it.modelId == null) {
                    Timber.e("toWrapper: ModelItem.modelId is null after reconstruction! cachedModelId: $modelId")
                    return null
                }

                ModelItemWrapper(
                    modelItem = it,
                    downloadedModelInfo = if (!isLocal && modelPath != null) {
                        // Only create DownloadedModelInfo if we have a valid modelPath
                        ChatDataManager.DownloadedModelInfo(
                            modelId, downloadTime, modelPath, lastChatTime
                        )
                    } else {
                        // For local models or non-local models without path, use null
                        null
                    },
                    downloadSize = downloadSize,
                    isPinned = isPinned,
                    sourceTag = getSourceLabel(context, modelId, isLocal)
                )
            }
        }
    }

    data class ModelListCache(
        @SerializedName("models") val models: List<ModelItemCacheDTO>,
        @SerializedName("timestamp") val timestamp: Long,
        @SerializedName("version") val version: Int = CACHE_VERSION
    )

    // ==================== PUBLIC API ====================

    init {
        applicationScope.launch {
            // Use MnnLlmApplication.getInstance() as context if not set yet
            val contextToUse = if (::context.isInitialized) {
                context
            } else {
                MnnLlmApplication.getInstance()
            }
            initialize(contextToUse)
        }
    }

    /**
     * Initialize ModelListManager with cache-then-network strategy
     * Called automatically on first access (observe or getCurrentModels)
     * Safe to call multiple times - will only initialize once
     */
    suspend fun initialize(context: Context) {
        loadingMutex.withLock {
            // Check if already initialized or currently initializing
            if (isInitialized || isInitializing) {
                Timber.d("Already initialized or initializing, skipping (initialized=$isInitialized, initializing=$isInitializing)")
                return
            }

            isInitializing = true

            try {
                this.context = context
                cacheFile = File(context.cacheDir, CACHE_FILE_NAME)
                startMarketDataSyncIfNeeded()

                // CRITICAL: Initialize TagMapper BEFORE loading/emitting cached data
                // This ensures tags display correctly from the first render
                try {
                    val marketConfig = ModelRepository.loadCachedOrAssets()
                    if (marketConfig != null) {
                        TagMapper.initializeFromConfig(marketConfig)
                        Timber.d("TagMapper initialized before cache emit: ${TagMapper.getAllTags().size} tags")
                    }
                } catch (e: Exception) {
                    Timber.w(e, "Failed to pre-initialize TagMapper")
                }

                // Handle builtin models before loading
                copyBuiltinModelsIfNeeded(context)

                // Step 1: Try to load and emit cached data first
                val loadedCachedModels = loadFromDiskCache()
                if (!loadedCachedModels.isNullOrEmpty()) {
                    Timber.d("Emitting cached data (${loadedCachedModels.size} models)")
                    _modelListState.value = ModelListState.Success(
                        models = loadedCachedModels,
                        source = DataSource.CACHE
                    )
                    loadedCachedModels.forEach {
                        modelIdModelMap[it.modelItem.modelId!!] = it.modelItem
                    }
                    hasEmittedInitialCache = true
                } else {
                    Timber.d("No valid cache found, showing loading state")
                    _modelListState.value = ModelListState.Loading
                }

                // Step 2: Immediately fetch fresh data in background
                try {
                    val freshModels = performModelLoading(context, "Initialize")

                    // Step 3: Compare and emit if different
                    if (hasDataChanged(loadedCachedModels, freshModels)) {
                        Timber.d("Fresh data differs from cache, emitting update (${freshModels.size} models)")
                        _modelListState.value = ModelListState.Success(
                            models = freshModels,
                            source = DataSource.FRESH
                        )

                        // Update memory cache
                        cachedModels = freshModels
                        modelIdModelMap.clear()
                        freshModels.forEach {
                            modelIdModelMap[it.modelItem.modelId!!] = it.modelItem
                        }

                        // Save to disk cache for next time
                        try {
                            saveToDiskCache(freshModels)
                        } catch (e: Exception) {
                            Timber.e(e, "Exception calling saveToDiskCache: ${e.message}")
                        }
                    } else {
                        // Update source to FRESH and ensure we transition out of Loading state
                        _modelListState.value = ModelListState.Success(
                            models = loadedCachedModels ?: freshModels,
                            source = DataSource.FRESH
                        )
                    }
                } catch (e: Exception) {
                    Timber.e(e, "Failed to load fresh data")
                    // If we already have cached data, don't emit error
                    if (_modelListState.value !is ModelListState.Success) {
                        _modelListState.value = ModelListState.Error(e)
                    }
                }

                isInitialized = true
                Timber.d("ModelListManager initialization complete")

            } finally {
                isInitializing = false
            }
        }
    }

    /**
     * Copy builtin models if they exist and haven't been copied yet
     */
    private suspend fun copyBuiltinModelsIfNeeded(context: Context) {
        try {
            if (!BuiltinModelManager.hasBuiltinModels(context)) {
                return
            }

            val copySuccess = BuiltinModelManager.ensureBuiltinModelsCopied(context) { _, _, _ -> }

            if (!copySuccess) {
                Timber.w("Failed to copy builtin models")
            }
        } catch (e: Exception) {
            Timber.w(e, "Error copying builtin models, continuing with initialization")
        }
    }

    /**
     * Subscribe to model list changes
     * Automatically triggers initialization on first call if context is set
     */
    fun observeModelList(): StateFlow<ModelListState> {
        // Auto-initialize if context is available and not yet initialized
        if (::context.isInitialized && !isInitialized && !isInitializing) {
            applicationScope.launch {
                initialize(context)
            }
        }
        return modelListState
    }

    /**
     * Subscribe to only successful model states
     */
    fun observeModels(): kotlinx.coroutines.flow.Flow<List<ModelItemWrapper>> = modelListState
        .filterIsInstance<ModelListState.Success>()
        .map { it.models }

    /**
     * Subscribe to models with data source tracking
     */
    fun observeModelsWithSource(): kotlinx.coroutines.flow.Flow<Pair<List<ModelItemWrapper>, DataSource>> =
        modelListState
            .filterIsInstance<ModelListState.Success>()
            .map { it.models to it.source }

    /**
     * Get current models synchronously (for non-reactive code)
     * Returns null if not yet initialized
     */
    fun getCurrentModels(): List<ModelItemWrapper>? {
        return (modelListState.value as? ModelListState.Success)?.models
    }

    /**
     * Ensure context is set for automatic initialization
     * Call this early in Application.onCreate() before any other access
     */
    fun setContext(context: Context) {
        if (!::context.isInitialized) {
            this.context = context.applicationContext
        }
    }

    /**
     * Check if current data is from cache
     */
    fun isShowingCachedData(): Boolean {
        return (modelListState.value as? ModelListState.Success)?.source == DataSource.CACHE
    }

    /**
     * Trigger model list refresh
     */
    suspend fun notifyModelListMayChange(reason: ChangeReason) {
        refreshModelList()
    }

    /**
     * Get the model map for external use
     */
    fun getModelIdModelMap(): Map<String, ModelItem> {
        return modelIdModelMap
    }

    /**
     * Get model tags for a specific model by modelId
     * Uses Flow-based cache for immediate access
     */
    fun getModelTags(modelId: String): List<String> {
        // First check memory map for immediate access
        val modelItem = modelIdModelMap[modelId]
        if (modelItem != null) {
            return modelItem.getTags()
        }

        // Never block callers (especially UI thread). Use current state snapshot only.
        val successState = modelListState.value as? ModelListState.Success ?: return emptyList()
        val wrapper = successState.models.find { it.modelItem.modelId == modelId }
        if (wrapper != null) {
            return wrapper.modelItem.getTags()
        }

        // Compensation: asynchronously warm up model list data for subsequent reads.
        scheduleTagWarmup(modelId)
        return emptyList()
    }

    private fun scheduleTagWarmup(modelId: String) {
        val runningJob = tagWarmupJob
        if (runningJob != null && runningJob.isActive) {
            return
        }
        tagWarmupJob = applicationScope.launch {
            try {
                if (!::context.isInitialized) {
                    return@launch
                }
                if (!isInitialized && !isInitializing) {
                    initialize(context)
                    return@launch
                }
                if (modelListState.value !is ModelListState.Success) {
                    refreshModelList()
                }
            } catch (e: Exception) {
                Timber.w(e, "scheduleTagWarmup failed for modelId=$modelId")
            }
        }
    }

    /**
     * Get extra tags for a specific model by modelId (not shown to users)
     */
    fun getExtraTags(modelId: String): List<String> {
        val modelItem = modelIdModelMap[modelId]
        if (modelItem == null) {
            return emptyList()
        }

        // Prefer framework-provided extra tags when available.
        val frameworkExtraTags = modelItem.getExtraTags()
        if (frameworkExtraTags.isNotEmpty()) {
            return frameworkExtraTags
        }

        // Fallback for local/legacy ModelItem implementations where getExtraTags() returns empty.
        return (modelItem.modelMarketItem as? ModelMarketItem)?.extraTags ?: emptyList()
    }

    /**
     * Check if a model is a thinking model by examining its tags
     */
    fun isThinkingModel(modelId: String): Boolean {
        val tags = getModelTags(modelId)
        return ModelTypeUtils.isThinkingModelByTags(tags)
    }

    /**
     * Check if a model is a visual model by examining its tags
     */
    fun isVisualModel(modelId: String): Boolean {
        val tags = getModelTags(modelId)
        return ModelTypeUtils.isVisualModelByTags(tags)
    }

    /**
     * Check if a model is a video model by examining its tags
     */
    fun isVideoModel(modelId: String): Boolean {
        val tags = getModelTags(modelId)
        return ModelTypeUtils.isVideoModelByTags(tags)
    }

    /**
     * Check if a model is an audio model by examining its tags
     */
    fun isAudioModel(modelId: String): Boolean {
        val tags = getModelTags(modelId)
        return ModelTypeUtils.isAudioModelByTags(tags)
    }

    // ==================== PRIVATE IMPLEMENTATION ====================

    /**
     * Internal refresh logic with disk caching
     */
    private suspend fun refreshModelList() {
        loadingMutex.withLock {
            // Don't show loading if we already have data (smooth update)
            if (_modelListState.value !is ModelListState.Success) {
                _modelListState.value = ModelListState.Loading
            }

            try {
                val oldModels = (_modelListState.value as? ModelListState.Success)?.models
                val freshModels = performModelLoading(context, "FlowRefresh")

                // Check if data actually changed
                if (hasDataChanged(oldModels, freshModels)) {
                    Timber.d("Model list changed, emitting update")
                    _modelListState.value = ModelListState.Success(
                        models = freshModels,
                        source = DataSource.FRESH
                    )

                    // Update memory cache
                    cachedModels = freshModels
                    modelIdModelMap.clear()
                    freshModels.forEach {
                        modelIdModelMap[it.modelItem.modelId!!] = it.modelItem
                    }

                    // Save to disk cache
                    Timber.d("About to call saveToDiskCache with ${freshModels.size} models (from refreshModelList)")
                    try {
                        saveToDiskCache(freshModels)
                    } catch (e: Exception) {
                        Timber.e(e, "Exception calling saveToDiskCache: ${e.message}")
                    }

                    _refreshEvents.emit(RefreshEvent.Success)
                } else {
                    _refreshEvents.emit(RefreshEvent.NoChange)
                }
            } catch (e: Exception) {
                Timber.e(e, "Failed to refresh model list")
                // Only emit error if we don't have valid data
                if (_modelListState.value !is ModelListState.Success) {
                    _modelListState.value = ModelListState.Error(e)
                }
                _refreshEvents.emit(RefreshEvent.Failed(e))
            }
        }
    }

    /**
     * Save models to disk cache using Gson
     */
    private suspend fun saveToDiskCache(models: List<ModelItemWrapper>) =
        withContext(Dispatchers.IO) {
            try {
                val cacheDTOs = models.mapNotNull { ModelItemCacheDTO.from(it) }
                val cache = ModelListCache(
                    models = cacheDTOs,
                    timestamp = System.currentTimeMillis()
                )
                val json = gson.toJson(cache)
                cacheFile.writeText(json)
            } catch (e: Exception) {
                Timber.e(e, "saveToDiskCache failed: ${e.message}")
                throw e
            }
        }

    /**
     * Load models from disk cache using Gson
     */
    private suspend fun loadFromDiskCache(): List<ModelItemWrapper>? = withContext(Dispatchers.IO) {
        try {
            if (!cacheFile.exists()) {
                return@withContext null
            }

            val json = cacheFile.readText()
            val cache = gson.fromJson(json, ModelListCache::class.java)

            // Validate cache version
            if (cache.version != CACHE_VERSION) {
                cacheFile.delete()
                return@withContext null
            }

            // Check cache age
            val cacheAge = System.currentTimeMillis() - cache.timestamp
            if (cacheAge > TimeUnit.DAYS.toMillis(MAX_CACHE_AGE_DAYS)) {
                cacheFile.delete()
                return@withContext null
            }

            // Reconstruct wrappers from cache
            val models = cache.models.mapNotNull { dto ->
                try {
                    dto.toWrapper(context)
                } catch (e: Exception) {
                    Timber.w(e, "Failed to reconstruct model ${dto.modelId}")
                    null
                }
            }
            val filteredModels = filterPrimaryListModels(models)

            if (filteredModels.isEmpty() && cache.models.isNotEmpty()) {
                return@withContext null
            }

            filteredModels
        } catch (e: Exception) {
            Timber.w(e, "Failed to load model list from disk cache")
            // Delete corrupted cache
            try {
                cacheFile.delete()
            } catch (deleteException: Exception) {
                Timber.w(deleteException, "Failed to delete corrupted cache")
            }
            null
        }
    }

    /**
     * Compare two model lists to detect changes
     */
    private fun hasDataChanged(
        cached: List<ModelItemWrapper>?,
        fresh: List<ModelItemWrapper>
    ): Boolean {
        if (cached == null) return true
        if (cached.size != fresh.size) return true

        // Compare model IDs and key properties
        val cachedIds = cached.map { it.modelItem.modelId }.toSet()
        val freshIds = fresh.map { it.modelItem.modelId }.toSet()

        if (cachedIds != freshIds) {
            Timber.d("Model IDs changed: removed=${cachedIds - freshIds}, added=${freshIds - cachedIds}")
            return true
        }

        // Compare properties that affect UI (pin status, download size, etc.)
        cached.zip(fresh).forEach { (c, f) ->
            if (c.isPinned != f.isPinned ||
                c.downloadSize != f.downloadSize ||
                c.lastChatTime != f.lastChatTime ||
                c.modelItem.modelName != f.modelItem.modelName ||
                c.modelItem.getTags() != f.modelItem.getTags()
            ) {
                Timber.d("Model properties changed for ${c.modelItem.modelId}")
                return true
            }
        }

        return false
    }

    /**
     * Clear cached models when models are downloaded or deleted
     * This will force the next load to reload from disk
     */
    fun clearModelCache() {
        synchronized(this) {
            cachedModels = null
            Timber.d("Model cache cleared - next load will reload from disk")
        }
    }

    /**
     * Execute the actual model loading logic (same as before)
     */
    private suspend fun performModelLoading(
        context: Context,
        caller: String
    ): List<ModelItemWrapper> {
        val modelWrappers = mutableListOf<ModelItemWrapper>()
        try {
            val chatDataManager = ChatDataManager.getInstance(context)
            val downloadedModels = mutableListOf<ChatDataManager.DownloadedModelInfo>()
            val pinnedModels = PreferenceUtils.getPinnedModels(context)

             // Add local models from LocalModelsProvider
             val localModels = LocalModelsProvider.getLocalModels()
             localModels.forEach { localModel ->
                 if (localModel.modelId != null && localModel.localPath != null) {
                     try {
                         val wrapper = processLocalModelWrapper(localModel, pinnedModels, context)
                         if (wrapper != null) {
                             modelWrappers.add(wrapper)
                         }
                     } catch (e: Exception) {
                         Timber.w(e, "Failed to process local model ${localModel.modelId}")
                     }
                 }
             }

            // Initialize ModelMarketCache and wait for it to be ready using Flow
            try {
                kotlinx.coroutines.withTimeout(2000) {
                    ModelMarketCache.observeModelMarketConfig(context).first()
                }
            } catch (e: Exception) {
                Timber.w(e, "Failed to initialize ModelMarketCache")
            }

            // Load downloaded models from .mnnmodels directory
            val mnnModelsDir = File(context.filesDir, ".mnnmodels")

            if (mnnModelsDir.exists() && mnnModelsDir.isDirectory) {
                scanModelsDirectory(mnnModelsDir, context, chatDataManager, downloadedModels)
            }

            // Convert downloaded models to wrapper format
            downloadedModels.forEach { downloadedModel ->
                try {
                    val wrapper = processModelWrapper(downloadedModel, pinnedModels, context)
                    if (wrapper != null) {
                        modelWrappers.add(wrapper)
                    }
                } catch (e: Exception) {
                    Timber.w(e, "Failed to process model ${downloadedModel.modelId}")
                }
            }

            val filteredModels = filterPrimaryListModels(modelWrappers)

            // Sort models: pinned as first priority, then recently used first, then by download time
            val sortedModels = filteredModels.sortedWith(
                compareByDescending<ModelItemWrapper> {
                    if (it.isPinned) 1 else 0
                }.thenByDescending {
                    if (it.lastChatTime > 0) 1 else 0
                }.thenByDescending {
                    if (it.lastChatTime > 0) it.lastChatTime else 0L
                }.thenByDescending {
                    if (it.isLocal) 1 else 0
                }.thenByDescending {
                    if (it.lastChatTime <= 0) it.downloadTime else 0L
                }
            )
            
            // Clear and cache modelId model to a map
                modelIdModelMap.clear()
                sortedModels.forEach {
                    modelIdModelMap[it.modelItem.modelId!!] = it.modelItem
                }

            return sortedModels
        } catch (e: Exception) {
            Timber.e(e, "loadAvailableModels: ERROR occurred (caller: $caller)")
            e.printStackTrace()
            return emptyList()
        }
    }

    /**
     * Process a single model wrapper with caching optimizations
     */
    private suspend fun processModelWrapper(
        downloadedModel: ChatDataManager.DownloadedModelInfo,
        pinnedModels: Set<String>,
        context: Context
    ): ModelItemWrapper? = withContext(Dispatchers.IO) {
        try {
            val modelItem = ModelItem.fromDownloadModel(
                context,
                downloadedModel.modelId,
                downloadedModel.modelPath
            )
            val marketItem = resolveMarketItemForModel(modelItem.modelId!!, modelItem.modelName)
            if (marketItem != null) {
                modelItem.modelMarketItem = marketItem
                modelItem.tags = marketItem.tags
            }

            // Ensure modelName is set
            if (modelItem.modelName.isNullOrEmpty()) {
                if (marketItem?.modelName != null) {
                    modelItem.modelName = marketItem.modelName
                } else {
                    modelItem.modelName = ModelUtils.getModelName(downloadedModel.modelId)
                }
            }

            // JOIN_POINT: Add builtin tag logic here
            // JOIN_POINT: Add builtin tag logic here
            if (modelItem.isBuiltin || modelItem.modelId?.startsWith("Builtin/") == true) {
                 val currentTags = modelItem.tags?.toMutableList() ?: mutableListOf()
                 if (!currentTags.contains("builtin")) {
                     currentTags.add(0, "builtin")
                 }
                 modelItem.tags = currentTags
            }
            // Get file size from saved data instead of calculating
            val downloadSize = try {
                val savedSize =
                    DownloadPersistentData.getDownloadSizeSaved(context, downloadedModel.modelId)
                if (savedSize > 0) {
                    savedSize
                } else {
                    // Fallback to file calculation if not saved
                    val file = File(downloadedModel.modelPath)
                    if (file.exists()) {
                        val calculatedSize = FileUtils.getFileSize(file)
                        // Save the calculated size for future use
                        DownloadPersistentData.saveDownloadSizeSaved(
                            context,
                            downloadedModel.modelId,
                            calculatedSize
                        )
                        calculatedSize
                    } else 0L
                }
            } catch (e: Exception) {
                0L
            }

            val isPinned = pinnedModels.contains(downloadedModel.modelId)
            return@withContext ModelItemWrapper(
                modelItem = modelItem,
                downloadedModelInfo = downloadedModel,
                downloadSize = downloadSize,
                isPinned = isPinned,
                sourceTag = getSourceLabel(context, modelItem.modelId, false)
            )
        } catch (e: Exception) {
            Timber.w(e, "Failed to process model ${downloadedModel.modelId}")
            return@withContext null
        }
    }

    /**
     * Process a single local model wrapper
     */
    private suspend fun processLocalModelWrapper(
        localModel: ModelItem,
        pinnedModels: Set<String>,
        context: Context
    ): ModelItemWrapper? = withContext(Dispatchers.IO) {
        try {
            // Check if config path exists for this local model
            val configPath = ModelUtils.getConfigPathForModel(localModel)
            android.util.Log.d("ModelListManager", "Processing local model: ${localModel.modelId}, configPath: $configPath")
            if (configPath == null || !File(configPath).exists()) {
                android.util.Log.w("ModelListManager", "Skipping local model ${localModel.modelId} due to missing config: $configPath")
                return@withContext null
            }

            // Load market tags for local model
//            localModel.loadMarketTags(context)

            val marketItem = resolveMarketItemForModel(localModel.modelId!!, localModel.modelName)
            if (marketItem != null) {
                localModel.modelMarketItem = marketItem
                localModel.tags = marketItem.tags
            }


            // JOIN_POINT: Add local tag logic here
            val currentTags = localModel.tags?.toMutableList() ?: mutableListOf()
            if (!currentTags.contains("local")) {
                currentTags.add(0, "local")
            }
            localModel.tags = currentTags

            // Ensure modelName is set
            if (localModel.modelName.isNullOrEmpty()) {
                if (marketItem?.modelName != null) {
                    localModel.modelName = marketItem.modelName
                } else {
                     // Fallback to directory name
                     localModel.modelName = ModelUtils.getModelName(localModel.modelId)
                }
            }

            // Get local model size from saved data or calculate
            val localSize = try {
                val savedSize =
                    DownloadPersistentData.getDownloadSizeSaved(context, localModel.modelId!!)
                if (savedSize > 0) {
                    savedSize
                } else {
                    // Fallback to file calculation if not saved
                    val file = File(localModel.localPath!!)
                    if (file.exists()) {
                        val calculatedSize = FileUtils.getFileSize(file)
                        // Save the calculated size for future use
                        DownloadPersistentData.saveDownloadSizeSaved(
                            context,
                            localModel.modelId!!,
                            calculatedSize
                        )
                        calculatedSize
                    } else 0L
                }
            } catch (e: Exception) {
                0L
            }

            val isPinned = pinnedModels.contains(localModel.modelId)

            return@withContext ModelItemWrapper(
                modelItem = localModel,
                downloadedModelInfo = null, // Local models don't have download info
                downloadSize = localSize,
                isPinned = isPinned,
                sourceTag = null // local models have no source tag
            )
        } catch (e: Exception) {
            Timber.w(e, "Failed to process local model ${localModel.modelId}")
            return@withContext null
        }
    }

    /**
     * Scan models directory recursively
     */
    private suspend fun scanModelsDirectory(
        directory: File,
        context: Context,
        chatDataManager: ChatDataManager,
        downloadedModels: MutableList<ChatDataManager.DownloadedModelInfo>,
        logger: StringBuilder? = null,
        dryRun: Boolean = false
    ) {
        val indent = if (logger != null) "  ".repeat(directory.absolutePath.split(File.separator).size - (context.filesDir.absolutePath.split(File.separator).size + 1)) else ""
        logger?.append("${indent}Scanning Dir: ${directory.name}\n")
        
        try {
            val files = directory.listFiles()
            if (files == null) {
                logger?.append("${indent}[EMPTY/ACCESSIBLE?]\n")
                return
            }

            // Collect all model IDs first to batch database queries
            val modelIds = mutableListOf<String>()
            val missingConfigModelIds = mutableListOf<String>()
            val modelPaths = mutableMapOf<String, String>()
            val modelTypes = mutableMapOf<String, String>()

            files.forEach { file ->
                val isBuiltinModel = directory.name == "builtin"
                
                // Logic 0: Skip HuggingFace cache directories (format: models--{org}--{repo})
                // These are HuggingFace Hub cache container directories, not actual models
                if (file.isDirectory && file.name.startsWith("models--")) {
                    logger?.append("${indent}[SKIP HF CACHE] ${file.name}\n")
                    return@forEach
                }

                // Logic 1: Priority Recurse
                // This must be checked BEFORE processing as a model to ensure we don't treat "modelscope" as a model and stop recursing
                if (file.isDirectory && (file.name == "modelscope" || file.name == "modelers" || file.name == "builtin")) {
                    logger?.append("${indent}[RECURSE CONTAINER] ${file.name}\n")
                    scanModelsDirectory(file, context, chatDataManager, downloadedModels, logger, dryRun)
                    return@forEach
                }

                // Logic 2: Allow both Symbolic Links AND Directories for downloaded models
                val shouldProcessFile = if (isBuiltinModel) {
                    file.isDirectory
                } else {
                    Files.isSymbolicLink(file.toPath()) || file.isDirectory
                }

                var status = "[IGNORED]"
                
                if (shouldProcessFile) {
                    val modelPath = file.absolutePath
                    val modelId = createModelIdFromPath(context, modelPath)
                    
                    if (modelId != null) {
                        // Check if config path exists for this model
                        // We strictly check the current directory, avoiding reliance on ModelDownloadManager
                        // because we are in the process of discovering models that might be unknown to the system.
                        val configFile = File(file, "config.json")
                        val configExists = configFile.exists()
                        val configPath = if (configExists) configFile.absolutePath else null
                        
                        if (configExists) {
                             status = "[MODEL CANDIDATE] ID=$modelId"
                             // Collect model info for batch processing
                             modelIds.add(modelId)
                             modelPaths[modelId] = modelPath
                             modelTypes[modelId] = if (isBuiltinModel) MODEL_TYPE_LLM else MODEL_TYPE_UNKNOWN
                        } else {
                             if (isBuiltinModel) {
                                 status = "[SKIPPED] No Config ($configPath)"
                             } else {
                                 // For downloaded models, allow DB marker based fallback (e.g. diffusion models).
                                 status = "[PENDING TYPE CHECK] ID=$modelId No Config"
                                 missingConfigModelIds.add(modelId)
                                 modelPaths[modelId] = modelPath
                             }
                        }
                    } else {
                         status = "[SKIPPED] Invalid Model Path"
                    }
                }
                
                logger?.append("${indent}- ${file.name} (Dir=${file.isDirectory}, Link=${Files.isSymbolicLink(file.toPath())}) -> $status\n")
            }

            // Batch process all collected models
            if (modelIds.isNotEmpty() || missingConfigModelIds.isNotEmpty()) {
                // Get model types for non-builtin models
                val nonBuiltinModels = modelIds.filter { modelTypes[it] == MODEL_TYPE_UNKNOWN }
                if (nonBuiltinModels.isNotEmpty()) {
                    withContext(Dispatchers.Main) {
                        nonBuiltinModels.forEach { modelId ->
                            try {
                                val modelType = chatDataManager.getDownloadModelType(modelId)
                                logger?.append("${indent}  > Check DB Type for $modelId: $modelType\n")
                                if (modelType != null) {
                                    modelTypes[modelId] = modelType
                                }
                            } catch (e: Exception) {
                                logger?.append("${indent}  > Error checking DB type for $modelId: ${e.message}\n")
                                Timber.w(e, "Failed to get model type for $modelId")
                            }
                        }
                    }
                }

                // For missing-config directories, include only diffusion models.
                if (missingConfigModelIds.isNotEmpty()) {
                    withContext(Dispatchers.Main) {
                        missingConfigModelIds.forEach { modelId ->
                            try {
                                val modelType = chatDataManager.getDownloadModelType(modelId)
                                logger?.append("${indent}  > Check Missing-Config DB Type for $modelId: $modelType\n")
                                if (shouldIncludeMissingConfigModelForScan(modelId, modelType)) {
                                    modelIds.add(modelId)
                                    modelTypes[modelId] = resolveModelTypeForDownloadHistory(modelId, modelType)
                                    logger?.append("${indent}    -> Included missing-config model: $modelId (type=${modelTypes[modelId]})\n")
                                } else {
                                    logger?.append("${indent}    -> Skipped missing-config model: $modelId\n")
                                }
                            } catch (e: Exception) {
                                logger?.append("${indent}  > Error checking missing-config type for $modelId: ${e.message}\n")
                                Timber.w(e, "Failed to check missing-config type for $modelId")
                            }
                        }
                    }
                }

                // Keep LLM and DIFFUSION. Keep UNKNOWN only when it doesn't look like a voice-only model.
                val llmModels = modelIds.filter { modelId ->
                    val modelType = modelTypes[modelId]
                    when {
                        modelType.equals(MODEL_TYPE_LLM, ignoreCase = true) -> true
                        modelType.equals(MODEL_TYPE_DIFFUSION, ignoreCase = true) -> true
                        modelType.equals(MODEL_TYPE_UNKNOWN, ignoreCase = true) -> !isLikelyVoiceModel(modelId)
                        else -> false
                    }
                }
                logger?.append("${indent}  > Filtering: ${modelIds.size} candidates -> ${llmModels.size} valid (LLM + DIFFUSION + UNKNOWN(non-voice))\n")

                if (llmModels.isNotEmpty()) {
                    // Batch get download times and last chat times
                    val downloadTimes = mutableMapOf<String, Long>()
                    val lastChatTimes = mutableMapOf<String, Long>()

                    withContext(Dispatchers.Main) {
                        llmModels.forEach { modelId ->
                            try {
                                val downloadTime = chatDataManager.getDownloadTime(modelId)
                                downloadTimes[modelId] = downloadTime
                                logger?.append("${indent}  > DB Download Time for $modelId: $downloadTime\n")

                                val lastChatTime = chatDataManager.getLastChatTime(modelId)
                                lastChatTimes[modelId] = lastChatTime
                                logger?.append("${indent}  > DB Last Chat Time for $modelId: $lastChatTime\n")

                                // Record download history if needed
                                if (downloadTime <= 0) {
                                    val modelTypeForHistory = resolveModelTypeForDownloadHistory(modelId, modelTypes[modelId])
                                    logger?.append("${indent}  > [${if (dryRun) "DRY RUN" else "ACTION"}] Recording download history for $modelId type=$modelTypeForHistory\n")
                                    if (!dryRun) {
                                        chatDataManager.recordDownloadHistory(
                                            modelId,
                                            modelPaths[modelId]!!,
                                            modelTypeForHistory
                                        )
                                    }
                                }
                            } catch (e: Exception) {
                                logger?.append("${indent}  > Error getting DB info for $modelId: ${e.message}\n")
                                Timber.w(e, "Failed to get database info for $modelId")
                            }
                        }
                    }

                    // Create DownloadedModelInfo objects
                    llmModels.forEach { modelId ->
                        val modelPath = modelPaths[modelId]!!
                        val file = File(modelPath)
                        val downloadTime = downloadTimes[modelId] ?: file.lastModified()
                        val lastChatTime = lastChatTimes[modelId] ?: 0L

                        if (dryRun) {
                             logger?.append("${indent}  > [DRY RUN] Would Add Model Output: $modelId\n")
                        } else {
                            downloadedModels.add(
                                ChatDataManager.DownloadedModelInfo(
                                    modelId, downloadTime, modelPath, lastChatTime
                                )
                            )
                        }
                    }
                }
            }
        } catch (e: Exception) {
            logger?.append("${indent}Scan Error: ${e.message}\n")
            Timber.e(e, "Error occurred while scanning ${directory.absolutePath}")
        }
    }
    
    /**
     * Debug method using the exact same logic as the real scan
     */
    suspend fun debugScanModels(context: Context): String {
        val sb = StringBuilder()
        sb.append("=== ModelListManager Debug Scan (Dry Run) ===\n")
        try {
            val directory = File(context.filesDir, ".mnnmodels")
            sb.append("Root: ${directory.absolutePath}\n")
            
            if (!directory.exists()) {
                sb.append("Root directory does not exist!\n")
                return sb.toString()
            }

            val chatDataManager = ChatDataManager.getInstance(context)
            val dummyList = mutableListOf<ChatDataManager.DownloadedModelInfo>()
            
            scanModelsDirectory(directory, context, chatDataManager, dummyList, sb, dryRun = true)
            
        } catch (e: Exception) {
            sb.append("Debug Scan Error: ${e.message}\n")
            Timber.e(e, "Debug scan failed")
        }
        return sb.toString()
    }

    private fun startMarketDataSyncIfNeeded() {
        if (Build.FINGERPRINT.contains("robolectric", ignoreCase = true)) {
            return
        }
        if (marketDataSyncJob != null) {
            return
        }
        marketDataSyncJob = marketSyncScope.launch {
            ModelRepository.marketDataFlow
                .filterNotNull()
                .collect { marketData ->
                    val currentSyncKey = buildMarketSyncKey(marketData.version)
                    if (!isInitialized || isInitializing) {
                        Timber.d(
                            "Market data changed while model list not ready: ${lastSyncedMarketDataKey} -> $currentSyncKey"
                        )
                        return@collect
                    }
                    if (currentSyncKey == lastSyncedMarketDataKey) {
                        return@collect
                    }
                    val previousSyncKey = lastSyncedMarketDataKey
                    lastSyncedMarketDataKey = currentSyncKey
                    try {
                        Timber.d("Market data changed: $previousSyncKey -> $currentSyncKey, refreshing model list")
                        notifyModelListMayChange(ChangeReason.MARKET_DATA_UPDATED)
                    } catch (e: Exception) {
                        Timber.w(e, "Failed to refresh model list for market version change")
                    }
                }
        }
    }

    private fun buildMarketSyncKey(version: String): String {
        val env = if (::context.isInitialized) {
            ModelRepository.getMarketEnvironment(context)
        } else {
            "unknown"
        }
        return "$env:$version"
    }

    private suspend fun resolveMarketItemForModel(
        modelId: String,
        modelNameHint: String?
    ): ModelMarketItem? {
        val exact = ModelMarketCache.getModelFromCache(modelId)
        if (exact != null) {
            return exact
        }

        val allMarketItems = ModelMarketCache.getAllCachedModels().values
        if (allMarketItems.isEmpty()) {
            return null
        }

        val expectedSuffix = normalizeKey(modelId.substringAfterLast('/'))
        if (expectedSuffix.isNotEmpty()) {
            val bySuffix = allMarketItems.firstOrNull {
                normalizeKey(it.modelId.substringAfterLast('/')) == expectedSuffix
            }
            if (bySuffix != null) {
                return bySuffix
            }
        }

        val expectedName = normalizeKey(modelNameHint ?: ModelUtils.getModelName(modelId))
        if (expectedName.isNotEmpty()) {
            return allMarketItems.firstOrNull { normalizeKey(it.modelName) == expectedName }
        }

        return null
    }

    private fun normalizeKey(value: String?): String {
        return value
            ?.trim()
            ?.lowercase(Locale.ROOT)
            ?.replace("_", "")
            ?.replace("-", "")
            ?.replace(" ", "")
            ?: ""
    }

    private fun filterPrimaryListModels(models: List<ModelItemWrapper>): List<ModelItemWrapper> {
        if (models.isEmpty()) {
            return models
        }
        val filtered = models.filterNot { wrapper ->
            shouldExcludeFromPrimaryList(wrapper.modelItem)
        }
        if (filtered.size != models.size) {
            val removedIds = models.asSequence()
                .mapNotNull { it.modelItem.modelId }
                .filter { modelId -> filtered.none { it.modelItem.modelId == modelId } }
                .toList()
            Timber.d("Filtered out non-LLM models from primary list: $removedIds")
        }
        return filtered
    }

    private fun shouldExcludeFromPrimaryList(modelItem: ModelItem): Boolean {
        val modelId = modelItem.modelId ?: return false
        val tags = modelItem.getTags()
        if (ModelTypeUtils.isAsrModelByTags(tags) || ModelTypeUtils.isTtsModelByTags(tags)) {
            return true
        }

        // Keep local debug/dev models visible by default. Only explicit ASR/TTS tags can hide them.
        if (modelId.startsWith("local/")) {
            return false
        }

        val modelName = modelItem.modelName ?: ModelUtils.getModelName(modelId).orEmpty()
        if (ModelTypeUtils.isTtsModel(modelName) || ModelTypeUtils.isTtsModel(modelId)) {
            return true
        }

        return isLikelyVoiceModel(modelId) || isLikelyVoiceModel(modelName)
    }

    private fun isLikelyVoiceModel(value: String?): Boolean {
        if (value.isNullOrBlank()) {
            return false
        }
        val normalized = value.lowercase(Locale.ROOT)
        return normalized.contains("tts") ||
            normalized.contains("asr") ||
            normalized.contains("bert-vits") ||
            normalized.contains("sherpa") ||
            normalized.contains("whisper") ||
            normalized.contains("zipformer")
    }

    internal fun shouldIncludeMissingConfigModelForScan(
        modelId: String,
        modelType: String?
    ): Boolean {
        if (modelType.equals(MODEL_TYPE_DIFFUSION, ignoreCase = true)) {
            return true
        }
        return modelType == null && ModelTypeUtils.isDiffusionModel(modelId)
    }

    internal fun resolveModelTypeForDownloadHistory(
        modelId: String,
        modelType: String?
    ): String {
        if (modelType.equals(MODEL_TYPE_DIFFUSION, ignoreCase = true) ||
            ModelTypeUtils.isDiffusionModel(modelId)) {
            return MODEL_TYPE_DIFFUSION
        }
        return MODEL_TYPE_LLM
    }

    /**
     * Source label for My Models tag (ModelScope / HuggingFace / Modelers / Builtin).
     * Used when building ModelItemWrapper so the UI can show the tag without re-deriving from modelId.
     * Note: Do not use isLocal here — ModelItem.isLocal is true whenever localPath is set (including downloaded models).
     * Only user-picked local models have modelId starting with "local/".
     */
    private fun getSourceLabel(context: Context, modelId: String?, isLocal: Boolean): String? {
        if (modelId == null || modelId.isBlank() || modelId.startsWith("local/")) return null
        return when {
            modelId.startsWith("HuggingFace/") || modelId.contains("taobao-mnn") -> context.getString(R.string.huggingface)
            modelId.startsWith("ModelScope/") -> context.getString(R.string.modelscope)
            modelId.startsWith("Modelers/") -> context.getString(R.string.modelers)
            modelId.startsWith("Builtin/") -> context.getString(R.string.builtin)
            else -> null
        }
    }

    /**
     * Create model ID from path (same logic as ModelListPresenter)
     */
    private fun createModelIdFromPath(context: Context, absolutePath: String): String? {
        // Extract the relative path from .mnnmodels directory
        val mnnModelsPath = File(context.filesDir, ".mnnmodels").absolutePath
        if (!absolutePath.startsWith(mnnModelsPath)) {
            return null
        }
        val relativePath = absolutePath.substring(mnnModelsPath.length + 1)
        return when {
            relativePath.startsWith("modelers/") -> {
                val modelName = relativePath.substring("modelers/".length)
                if (modelName.isNotEmpty()) "Modelers/MNN/$modelName" else null
            }

            relativePath.startsWith("modelscope/") -> {
                val modelName = relativePath.substring("modelscope/".length)
                if (modelName.isNotEmpty()) "ModelScope/MNN/$modelName" else null
            }

            relativePath.startsWith("builtin/") -> {
                val modelName = relativePath.substring("builtin/".length)
                if (modelName.isNotEmpty()) "Builtin/MNN/$modelName" else null
            }

            !relativePath.contains("/") -> {
                if (relativePath.isNotEmpty()) "HuggingFace/taobao-mnn/$relativePath" else null
            }

            else -> null
        }
    }
}
