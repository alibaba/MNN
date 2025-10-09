package com.alibaba.mnnllm.android.modelist

import android.content.Context
import android.util.Log
import com.alibaba.mls.api.ModelItem
import com.alibaba.mnnllm.android.tag.ModelTagsCache
import com.alibaba.mnnllm.android.chat.model.ChatDataManager
import com.alibaba.mnnllm.android.model.ModelUtils
import com.alibaba.mnnllm.android.modelmarket.ModelRepository
import com.alibaba.mnnllm.android.modelmarket.ModelMarketItem
import com.alibaba.mls.api.download.DownloadPersistentData
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import java.io.File
import java.nio.file.Files
import com.alibaba.mnnllm.android.modelmarket.ModelMarketUtils
import com.alibaba.mnnllm.android.utils.FileUtils
import com.alibaba.mnnllm.android.utils.PreferenceUtils

object ModelListManager {
    private const val TAG = "ModelListManager"

    // Map to store modelId to ModelItem mapping for external access
    private val modelIdModelMap = mutableMapOf<String, ModelItem>()

    /**
     * Get the model map for external use
     */
    fun getModelIdModelMap(): Map<String, ModelItem> {
        return modelIdModelMap
    }

    /**
     * Get model tags for a specific model by modelId
     */
    fun getModelTags(modelId: String): List<String> {
        val modelItem = modelIdModelMap[modelId]
        Log.e(TAG, "getModelTags: $modelId modelItem: ${modelItem?.modelId} ${modelItem?.modelMarketItem?.modelName}")
        
        // If modelItem is found in map, return its tags
        if (modelItem != null) {
            return modelItem.getTags()
        }
        
        // Fallback: try to read tags directly from market_config.json
        Log.d(TAG, "ModelItem not found in map for $modelId, trying fallback to read market_config.json")
        return try {
            val marketItem = runBlocking { ModelMarketUtils.readMarketConfigFromLocal(modelId) }
            marketItem?.tags ?: emptyList()
        } catch (e: Exception) {
            Log.w(TAG, "Failed to read market config for $modelId in fallback", e)
            emptyList()
        }
    }

    /**
     * Get extra tags for a specific model by modelId (not shown to users)
     */
    fun getExtraTags(modelId: String): List<String> {
        val modelItem = modelIdModelMap[modelId]
        return modelItem?.getExtraTags() ?: emptyList()
    }

    /**
     * Check if a model is a thinking model by examining its tags
     */
    fun isThinkingModel(modelId: String): Boolean {
        val tags = getModelTags(modelId)
        return ModelUtils.isThinkingModelByTags(tags)
    }

    /**
     * Check if a model is a visual model by examining its tags
     */
    fun isVisualModel(modelId: String): Boolean {
        val tags = getModelTags(modelId)
        return ModelUtils.isVisualModelByTags(tags)
    }

    /**
     * Check if a model is an audio model by examining its tags
     */
    fun isAudioModel(modelId: String): Boolean {
        val tags = getModelTags(modelId)
        return ModelUtils.isAudioModelByTags(tags)
    }

    /**
     * Load all available models (downloaded + local) wrapped with their info
     */
    suspend fun loadAvailableModels(context: Context): List<ModelItemWrapper> = withContext(Dispatchers.IO) {
        val startTime = System.currentTimeMillis()
        Log.d(TAG, "loadAvailableModels: Starting optimized version")
        
        val modelWrappers = mutableListOf<ModelItemWrapper>()
        try {
            val step1Start = System.currentTimeMillis()
            val chatDataManager = ChatDataManager.getInstance(context)
            val downloadedModels = mutableListOf<ChatDataManager.DownloadedModelInfo>()
            val pinnedModels = PreferenceUtils.getPinnedModels(context)
            
            //Initialize ModelMarketUtils cache (unified memory cache)
            val marketDataStart = System.currentTimeMillis()
            try {
                ModelMarketUtils.initializeCache(context)
                val allMarketItems = ModelMarketUtils.getAllCachedModels()
                Log.d(TAG, "Loaded ${allMarketItems.size} market items from ModelMarketUtils cache")
            } catch (e: Exception) {
                Log.w(TAG, "Failed to initialize ModelMarketUtils cache", e)
            }
            val marketDataTime = System.currentTimeMillis() - marketDataStart
            
            val step1Time = System.currentTimeMillis() - step1Start
            Log.d(TAG, "loadAvailableModels: Step 1 (initialization + market data) took ${step1Time}ms (market data: ${marketDataTime}ms)")

            // Load downloaded models from .mnnmodels directory
            val step2Start = System.currentTimeMillis()
            Log.d(TAG, "loadAvailableModels: Starting Step 2 - scanning models directory")
            val mnnModelsDir = File(context.filesDir, ".mnnmodels")
            Log.d(TAG, "loadAvailableModels: .mnnmodels directory path: ${mnnModelsDir.absolutePath}")
            Log.d(TAG, "loadAvailableModels: .mnnmodels directory exists: ${mnnModelsDir.exists()}")
            Log.d(TAG, "loadAvailableModels: .mnnmodels directory isDirectory: ${mnnModelsDir.isDirectory}")
            
            if (mnnModelsDir.exists() && mnnModelsDir.isDirectory) {
                Log.d(TAG, "loadAvailableModels: Calling scanModelsDirectory")
                scanModelsDirectory(mnnModelsDir, context, chatDataManager, downloadedModels)
                Log.d(TAG, "loadAvailableModels: scanModelsDirectory completed")
            } else {
                Log.d(TAG, "loadAvailableModels: .mnnmodels directory not found or not a directory")
            }
            val step2Time = System.currentTimeMillis() - step2Start
            Log.d(TAG, "loadAvailableModels: Step 2 (scanModelsDirectory) took ${step2Time}ms, found ${downloadedModels.size} models")
            // Initialize tags cache for proper tag loading
            val step3Start = System.currentTimeMillis()
            ModelTagsCache.initializeCache(context)
            val step3Time = System.currentTimeMillis() - step3Start
            Log.d(TAG, "loadAvailableModels: Step 3 (ModelTagsCache.initializeCache) took ${step3Time}ms")

            // Convert downloaded models to wrapper format
            val step4Start = System.currentTimeMillis()
            Log.d(TAG, "loadAvailableModels: Starting Step 4 - processing ${downloadedModels.size} downloaded models")
            
            downloadedModels.forEach { downloadedModel ->
                try {
                    val wrapper = processModelWrapper(downloadedModel, pinnedModels, context)
                    if (wrapper != null) {
                        modelWrappers.add(wrapper)
                    }
                } catch (e: Exception) {
                    Log.w(TAG, "Failed to process model ${downloadedModel.modelId}", e)
                }
            }
            val step4Time = System.currentTimeMillis() - step4Start
            Log.d(TAG, "loadAvailableModels: Step 4 (convert downloaded models) took ${step4Time}ms")

            // Add local models from ModelUtils.localModelList
            val step5Start = System.currentTimeMillis()
            Log.d(TAG, "loadAvailableModels: Starting Step 5 - processing ${ModelUtils.localModelList.size} local models")
            
            ModelUtils.localModelList.forEach { localModel ->
                if (localModel.modelId != null && localModel.localPath != null) {
                    try {
                        val wrapper = processLocalModelWrapper(localModel, pinnedModels, context)
                        if (wrapper != null) {
                            modelWrappers.add(wrapper)
                        }
                    } catch (e: Exception) {
                        Log.w(TAG, "Failed to process local model ${localModel.modelId}", e)
                    }
                }
            }
            val step5Time = System.currentTimeMillis() - step5Start
            Log.d(TAG, "loadAvailableModels: Step 5 (process local models) took ${step5Time}ms")
            // Sort models: pinned as first priority, then recently used first, then by download time
            val step6Start = System.currentTimeMillis()
            val sortedModels = modelWrappers.sortedWith(
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
            val step6Time = System.currentTimeMillis() - step6Start
            Log.d(TAG, "loadAvailableModels: Step 6 (sorting) took ${step6Time}ms")
            // Clear and cache modelId model to a map
            val step7Start = System.currentTimeMillis()
            modelIdModelMap.clear()
            sortedModels.forEach {
                //add log for each it.modelItem.modelId
                modelIdModelMap[it.modelItem.modelId!!] = it.modelItem
            }
            val step7Time = System.currentTimeMillis() - step7Start
            Log.d(TAG, "loadAvailableModels: Step 7 (cache modelId map) took ${step7Time}ms")
            
            val totalTime = System.currentTimeMillis() - startTime
            Log.d(TAG, "loadAvailableModels: TOTAL TIME ${totalTime}ms - Found ${sortedModels.size} total models")
            Log.d(TAG, "loadAvailableModels: Performance breakdown - Init:${step1Time}ms, Scan:${step2Time}ms, Cache:${step3Time}ms, Convert:${step4Time}ms, Local:${step5Time}ms, Sort:${step6Time}ms, Map:${step7Time}ms")
            
            return@withContext sortedModels
        } catch (e: Exception) {
            Log.e(TAG, "loadAvailableModels: ERROR occurred", e)
            e.printStackTrace()
            return@withContext emptyList()
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
            val modelItem = ModelItem.fromDownloadModel(context, downloadedModel.modelId, downloadedModel.modelPath)
            
            // Use ModelMarketUtils cache instead of IO operation
            modelItem.modelMarketItem = ModelMarketUtils.getModelFromCache(downloadedModel.modelId)
            
            // Get file size from saved data instead of calculating
            val downloadSize = try {
                val savedSize = DownloadPersistentData.getDownloadSizeSaved(context, downloadedModel.modelId)
                if (savedSize > 0) {
                    savedSize
                } else {
                    // Fallback to file calculation if not saved
                    val file = File(downloadedModel.modelPath)
                    if (file.exists()) {
                        val calculatedSize = FileUtils.getFileSize(file)
                        // Save the calculated size for future use
                        DownloadPersistentData.saveDownloadSizeSaved(context, downloadedModel.modelId, calculatedSize)
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
                isPinned = isPinned
            )
        } catch (e: Exception) {
            Log.w(TAG, "Failed to process model ${downloadedModel.modelId}", e)
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
            if (configPath == null || !File(configPath).exists()) {
                return@withContext null
            }

            // Load market tags for local model
            localModel.loadMarketTags(context)
            
            // Use ModelMarketUtils cache instead of IO operation
            localModel.modelMarketItem = ModelMarketUtils.getModelFromCache(localModel.modelId!!)

            // Get local model size from saved data or calculate
            val localSize = try {
                val savedSize = DownloadPersistentData.getDownloadSizeSaved(context, localModel.modelId!!)
                if (savedSize > 0) {
                    savedSize
                } else {
                    // Fallback to file calculation if not saved
                    val file = File(localModel.localPath!!)
                    if (file.exists()) {
                        val calculatedSize = FileUtils.getFileSize(file)
                        // Save the calculated size for future use
                        DownloadPersistentData.saveDownloadSizeSaved(context, localModel.modelId!!, calculatedSize)
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
                isPinned = isPinned
            )
        } catch (e: Exception) {
            Log.w(TAG, "Failed to process local model ${localModel.modelId}", e)
            return@withContext null
        }
    }

    /**
     * Scan models directory recursively (same logic as ModelListPresenter)
     */
    private suspend fun scanModelsDirectory(
        directory: File, 
        context: Context,
        chatDataManager: ChatDataManager, 
        downloadedModels: MutableList<ChatDataManager.DownloadedModelInfo>
    ) {
        val scanStartTime = System.currentTimeMillis()
        Log.d(TAG, "scanModelsDirectory: START - scanning directory ${directory.absolutePath}")
        
        var fileCount = 0
        try {
            val files = directory.listFiles()
            fileCount = files?.size ?: 0
            Log.d(TAG, "scanModelsDirectory: found $fileCount files in ${directory.absolutePath}")
            
            if (files == null) {
                Log.w(TAG, "scanModelsDirectory: listFiles() returned null for ${directory.absolutePath}")
                return
            }
            
            Log.d(TAG, "scanModelsDirectory: files list: ${files.map { it.name }}")
        
        // Collect all model IDs first to batch database queries
        val modelIds = mutableListOf<String>()
        val modelPaths = mutableMapOf<String, String>()
        val modelTypes = mutableMapOf<String, String>()
        
            files.forEach { file ->
                val isBuiltinModel = directory.name == "builtin"
                val shouldProcessFile = if (isBuiltinModel) {
                    file.isDirectory
                } else {
                    Files.isSymbolicLink(file.toPath())
                }
                
                if (shouldProcessFile) {
                val modelPath = file.absolutePath
                val modelId = createModelIdFromPath(context, modelPath)
                Log.d(TAG, "Processing file: $file.name, isBuiltinModel: $isBuiltinModel, modelId: $modelId")
                if (modelId != null) {
                    // Check if config path exists for this model first (fast check)
                    val configPath = ModelUtils.getConfigPathForModel(modelId)
                    if (configPath == null || !File(configPath).exists()) {
                        return@forEach
                    }
                    
                    // Collect model info for batch processing
                    modelIds.add(modelId)
                    modelPaths[modelId] = modelPath
                    modelTypes[modelId] = if (isBuiltinModel) "LLM" else "UNKNOWN"
                }
        } else if (file.isDirectory && (file.name == "modelscope" || file.name == "modelers" || file.name == "builtin")) {
            scanModelsDirectory(file, context, chatDataManager, downloadedModels)
        }
        }
        
        // Batch process all collected models
        if (modelIds.isNotEmpty()) {
            val batchStartTime = System.currentTimeMillis()
            
            // Get model types for non-builtin models
            val nonBuiltinModels = modelIds.filter { modelTypes[it] == "UNKNOWN" }
            if (nonBuiltinModels.isNotEmpty()) {
                val modelTypeStart = System.currentTimeMillis()
                withContext(Dispatchers.Main) {
                    nonBuiltinModels.forEach { modelId ->
                        try {
                            val modelType = chatDataManager.getDownloadModelType(modelId)
                            if (modelType != null) {
                                modelTypes[modelId] = modelType
                            }
                        } catch (e: Exception) {
                            Log.w(TAG, "Failed to get model type for $modelId", e)
                        }
                    }
                }
                val modelTypeTime = System.currentTimeMillis() - modelTypeStart
                Log.d(TAG, "Batch model type queries took ${modelTypeTime}ms")
            }
            
            // Filter out non-LLM models
            val llmModels = modelIds.filter { modelTypes[it] == "LLM" }
            Log.d(TAG, "Found ${llmModels.size} LLM models out of ${modelIds.size} total models")
            
            // Batch get download times and last chat times
            val dbBatchStart = System.currentTimeMillis()
            val downloadTimes = mutableMapOf<String, Long>()
            val lastChatTimes = mutableMapOf<String, Long>()
            
            withContext(Dispatchers.Main) {
                llmModels.forEach { modelId ->
                    try {
                        val downloadTime = chatDataManager.getDownloadTime(modelId)
                        downloadTimes[modelId] = downloadTime
                        
                        val lastChatTime = chatDataManager.getLastChatTime(modelId)
                        lastChatTimes[modelId] = lastChatTime
                        
                        // Record download history if needed
                        if (downloadTime <= 0) {
                            chatDataManager.recordDownloadHistory(modelId, modelPaths[modelId]!!, "LLM")
                        }
                    } catch (e: Exception) {
                        Log.w(TAG, "Failed to get database info for $modelId", e)
                    }
                }
            }
            val dbBatchTime = System.currentTimeMillis() - dbBatchStart
            Log.d(TAG, "Batch database queries took ${dbBatchTime}ms")
            
            // Create DownloadedModelInfo objects
            llmModels.forEach { modelId ->
                val modelPath = modelPaths[modelId]!!
                val file = File(modelPath)
                val downloadTime = downloadTimes[modelId] ?: file.lastModified()
                val lastChatTime = lastChatTimes[modelId] ?: 0L
                
                downloadedModels.add(ChatDataManager.DownloadedModelInfo(
                    modelId, downloadTime, modelPath, lastChatTime
                ))
                
                Log.d(TAG, "Found LLM model with valid config: $modelId at $modelPath")
            }
            
            val batchTotalTime = System.currentTimeMillis() - batchStartTime
            Log.d(TAG, "Batch processing completed in ${batchTotalTime}ms")
        }
        
        } catch (e: Exception) {
            Log.e(TAG, "scanModelsDirectory: ERROR occurred while scanning ${directory.absolutePath}", e)
            throw e
        }
        
        val scanTotalTime = System.currentTimeMillis() - scanStartTime
        Log.d(TAG, "scanModelsDirectory: COMPLETED scanning ${directory.absolutePath} in ${scanTotalTime}ms, processed $fileCount files")
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