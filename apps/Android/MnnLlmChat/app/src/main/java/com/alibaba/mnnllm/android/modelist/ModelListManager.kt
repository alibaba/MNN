package com.alibaba.mnnllm.android.modelist

import android.content.Context
import android.util.Log
import com.alibaba.mls.api.ModelItem
import com.alibaba.mls.api.ModelTagsCache
import com.alibaba.mnnllm.android.chat.model.ChatDataManager
import com.alibaba.mnnllm.android.model.ModelUtils
import com.alibaba.mnnllm.android.modelmarket.ModelRepository
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.nio.file.Files
import com.alibaba.mnnllm.android.modelmarket.ModelMarketUtils
import com.alibaba.mnnllm.android.utils.FileUtils
import com.alibaba.mnnllm.android.utils.PreferenceUtils

object ModelListManager {
    private const val TAG = "ModelListManager"

    /**
     * Load all available models (downloaded + local) wrapped with their info
     */
    suspend fun loadAvailableModels(context: Context): List<ModelItemWrapper> = withContext(Dispatchers.IO) {
        val modelWrappers = mutableListOf<ModelItemWrapper>()
        try {
            val chatDataManager = ChatDataManager.getInstance(context)
            val downloadedModels = mutableListOf<ChatDataManager.DownloadedModelInfo>()
            val pinnedModels = PreferenceUtils.getPinnedModels(context)
            val modelRepository = ModelRepository(context)

            // Load downloaded models from .mnnmodels directory
            val mnnModelsDir = File(context.filesDir, ".mnnmodels")
            if (mnnModelsDir.exists() && mnnModelsDir.isDirectory) {
                scanModelsDirectory(mnnModelsDir, context, chatDataManager, downloadedModels, modelRepository)
            } else {
                Log.d(TAG, "loadAvailableModels: .mnnmodels directory not found")
            }
            Log.d(TAG, "loadAvailableModels scanModelsDirectory finished")
            // Initialize tags cache for proper tag loading
            ModelTagsCache.initializeCache(context)

            // Convert downloaded models to wrapper format
            downloadedModels.forEach { downloadedModel ->
                Log.d(TAG, "Found downloaded model: ${downloadedModel.modelId} at ${downloadedModel.modelPath}")

                val modelItem = ModelItem.fromDownloadModel(context, downloadedModel.modelId, downloadedModel.modelPath)
                // Set market item data if available
                modelItem.modelMarketItem = ModelMarketUtils.readMarketConfig(downloadedModel.modelId)

                // Calculate download size
                val downloadSize = try {
                    val file = File(downloadedModel.modelPath)
                    if (file.exists()) file.length() else 0L
                } catch (e: Exception) {
                    Log.w(TAG, "Failed to get file size for ${downloadedModel.modelPath}", e)
                    0L
                }

                val isPinned = pinnedModels.contains(downloadedModel.modelId)
                val modelItemWrapper = ModelItemWrapper(
                    modelItem = modelItem,
                    downloadedModelInfo = downloadedModel,
                    downloadSize = downloadSize,
                    isPinned = isPinned
                )
                modelWrappers.add(modelItemWrapper)
            }
            Log.d(TAG, "loadAvailableModels converted to wrapper format")

            // Add local models from ModelUtils.localModelList
            ModelUtils.localModelList.forEach { localModel ->
                if (localModel.modelId != null && localModel.localPath != null) {
                    // Check if config path exists for this local model
                    val configPath = ModelUtils.getConfigPathForModel(localModel)
                    if (configPath == null || !File(configPath).exists()) {
                        Log.d(TAG, "Skipping local model ${localModel.modelId}: config path not found or doesn't exist (path: $configPath)")
                        return@forEach
                    }

                    Log.d(TAG, "Found local model: ${localModel.modelId} at ${localModel.localPath}")

                    // Load market tags for local model
                    localModel.loadMarketTags(context)

                    // Calculate local model size
                    val localSize = try {
                        val file = File(localModel.localPath!!)
                        if (file.exists()) FileUtils.getFileSize(file) else 0L
                    } catch (e: Exception) {
                        Log.w(TAG, "Failed to get file size for ${localModel.localPath}", e)
                        0L
                    }

                    val isPinned = pinnedModels.contains(localModel.modelId)

                    modelWrappers.add(
                        ModelItemWrapper(
                        modelItem = localModel,
                        downloadedModelInfo = null, // Local models don't have download info
                        downloadSize = localSize,
                        isPinned = isPinned
                    )
                    )
                }
            }
            Log.d(TAG, "loadAvailableModels localModelList added")
            // Sort models: pinned as first priority, then recently used first, then by download time
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
            Log.d(TAG, "loadAvailableModels sort complte: Found ${sortedModels.size} total models")
            return@withContext sortedModels
        } catch (e: Exception) {
            Log.e(TAG, "Error loading available models", e)
            return@withContext emptyList()
        }
    }

    /**
     * Scan models directory recursively (same logic as ModelListPresenter)
     */
    private suspend fun scanModelsDirectory(
        directory: File, 
        context: Context,
        chatDataManager: ChatDataManager, 
        downloadedModels: MutableList<ChatDataManager.DownloadedModelInfo>,
        modelRepository: ModelRepository
    ) {
        directory.listFiles()?.forEach { file ->
            if (Files.isSymbolicLink(file.toPath())) {
                val modelPath = file.absolutePath
                val modelId = createModelIdFromPath(context, modelPath)
                if (modelId != null) {
                    // Check if this is an LLM model, skip if it's TTS or ASR
                    // First try to get model type from database, fallback to repository
                    // Database access on Main thread to avoid connection pool issues
                    val modelType = withContext(Dispatchers.Main) {
                        chatDataManager.getDownloadModelType(modelId)
                    } ?: runBlocking {
                        try {
                            modelRepository.getModelType(modelId)
                        } catch (e: Exception) {
                            Log.w(TAG, "Failed to get model type for $modelId, assuming LLM", e)
                            "LLM"
                        }
                    }
                    
                    if (modelType != "LLM") {
                        Log.d(TAG, "Skipping non-LLM model: $modelId (type: $modelType)")
                        return@forEach
                    }
                    
                    // Check if config path exists for this model
                    val modelName = ModelUtils.getModelName(modelId)
                    val configPath = ModelUtils.getConfigPathForModel(modelId)
                    if (configPath == null || !File(configPath).exists()) {
                        Log.d(TAG, "Skipping model $modelId: config path not found or doesn't exist (path: $configPath)")
                        return@forEach
                    }
                    
                    // Database access on Main thread
                    var downloadTime = withContext(Dispatchers.Main) {
                        chatDataManager.getDownloadTime(modelId)
                    }
                    if (downloadTime <= 0) {
                        downloadTime = file.lastModified()
                        try {
                            withContext(Dispatchers.Main) {
                                chatDataManager.recordDownloadHistory(modelId, modelPath, modelType)
                            }
                        } catch (e: Exception) {
                            Log.w(TAG, "Failed to record model in database: $modelId", e)
                        }
                    }
                    
                    // Database access on Main thread
                    var lastChatTime = withContext(Dispatchers.Main) {
                        chatDataManager.getLastChatTime(modelId)
                    }
                    Log.d(TAG, "getLastChatTime: $modelId lastChatTime: $lastChatTime")
                    val fixedModelId = "taobao-mnn/$modelName"
                    if (lastChatTime < 10000) {
                        lastChatTime = withContext(Dispatchers.Main) {
                            chatDataManager.getLastChatTime(fixedModelId)
                        }
                        Log.d(TAG, "getLastChatTimeFix: $fixedModelId lastChatTime: $lastChatTime")
                    }
                    if (lastChatTime < 10000) {
                        lastChatTime = withContext(Dispatchers.Main) {
                            chatDataManager.getLastChatTime(modelName!!)
                        }
                        Log.d(TAG, "getLastChatTimeFix: $modelName lastChatTime: $lastChatTime")
                    }
                    
                    downloadedModels.add(ChatDataManager.DownloadedModelInfo(
                        modelId, downloadTime, modelPath, lastChatTime
                    ))
                    
                    Log.d(TAG, "Found LLM model with valid config: $modelId at $modelPath (config: $configPath)")
                }
            } else if (file.isDirectory && (file.name == "modelscope" || file.name == "modelers")) {
                scanModelsDirectory(file, context, chatDataManager, downloadedModels, modelRepository)
            }
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
            !relativePath.contains("/") -> {
                if (relativePath.isNotEmpty()) "HuggingFace/taobao-mnn/$relativePath" else null
            }
            else -> null
        }
    }
}