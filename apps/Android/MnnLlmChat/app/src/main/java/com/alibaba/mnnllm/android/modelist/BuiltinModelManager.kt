package com.alibaba.mnnllm.android.modelist

import android.content.Context
import android.util.Log
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.utils.FileSplitter
import com.alibaba.mnnllm.android.utils.PreferenceUtils
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStream

/**
 * Manager for handling builtin models from assets
 */
object BuiltinModelManager {
    private const val TAG = "BuiltinModelManager"
    private const val ASSETS_BUILTIN_MODELS_DIR = "builtin_models"
    private const val BUILTIN_MODELS_COPIED_KEY = "builtin_models_copied_v1"
    
    /**
     * Check if there are any builtin models available in assets
     * @param context Android context
     * @return true if builtin models exist in assets, false otherwise
     */
    fun hasBuiltinModels(context: Context): Boolean {
        return try {
            val assetManager = context.assets
            val modelDirs = assetManager.list(ASSETS_BUILTIN_MODELS_DIR) ?: emptyArray()
            modelDirs.isNotEmpty()
        } catch (e: Exception) {
            Log.w(TAG, "Error checking for builtin models", e)
            false
        }
    }
    
    /**
     * Check if builtin models need to be copied and copy them if necessary
     * @param context Android context
     * @param onProgress Callback for progress updates (current, total, message)
     * @return true if copy was successful or not needed, false if failed
     */
    suspend fun ensureBuiltinModelsCopied(
        context: Context,
        onProgress: ((current: Int, total: Int, message: String) -> Unit)? = null
    ): Boolean = withContext(Dispatchers.IO) {
        try {
            // First check if there are any builtin models in assets
            if (!hasBuiltinModels(context)) {
                Log.d(TAG, "No builtin models found in assets, skipping copy process")
                return@withContext true
            }
            
            // Check if builtin models have already been copied
            if (PreferenceUtils.getBoolean(context, BUILTIN_MODELS_COPIED_KEY, false)) {
                // Double check that the builtin directory actually exists
                val builtinModelsDir = File(context.filesDir, ".mnnmodels/builtin")
                if (builtinModelsDir.exists() && builtinModelsDir.isDirectory) {
                    val modelDirs = builtinModelsDir.listFiles()?.filter { it.isDirectory }
                    if (!modelDirs.isNullOrEmpty()) {
                        Log.d(TAG, "Builtin models already copied and directory exists with ${modelDirs.size} models, skipping")
                        return@withContext true
                    } else {
                        Log.w(TAG, "Builtin directory exists but is empty, will re-copy")
                    }
                } else {
                    Log.w(TAG, "Builtin models marked as copied but directory doesn't exist, will re-copy")
                }
            }
            
            Log.d(TAG, "Starting builtin models copy process")
            onProgress?.invoke(0, 100, context.getString(R.string.copying_builtin_models))
            
            val assetManager = context.assets
            val builtinModelsDir = File(context.filesDir, ".mnnmodels/builtin")
            
            // Create builtin directory if it doesn't exist
            if (!builtinModelsDir.exists()) {
                builtinModelsDir.mkdirs()
                Log.d(TAG, "Created builtin models directory: ${builtinModelsDir.absolutePath}")
            }
            
            // List all models in assets/builtin_models
            val modelDirs = assetManager.list(ASSETS_BUILTIN_MODELS_DIR) ?: emptyArray()
            Log.d(TAG, "Found ${modelDirs.size} builtin models to copy: ${modelDirs.joinToString(", ")}")
            
            if (modelDirs.isEmpty()) {
                Log.w(TAG, "No builtin models found in assets")
                // Mark as copied even if empty to avoid repeated checks
                PreferenceUtils.setBoolean(context, BUILTIN_MODELS_COPIED_KEY, true)
                return@withContext true
            }
            
            var totalFiles = 0
            var copiedFiles = 0
            
            // First pass: count total files
            for (modelDir in modelDirs) {
                val modelAssetPath = "$ASSETS_BUILTIN_MODELS_DIR/$modelDir"
                totalFiles += countFilesInAssetDirectory(assetManager, modelAssetPath)
            }
            
            Log.d(TAG, "Total files to copy: $totalFiles")
            
            // Second pass: copy files
            for (modelDir in modelDirs) {
                val modelAssetPath = "$ASSETS_BUILTIN_MODELS_DIR/$modelDir"
                val modelLocalDir = File(builtinModelsDir, modelDir)
                
                Log.d(TAG, "Copying model: $modelDir")
                onProgress?.invoke(
                    (copiedFiles * 100) / totalFiles,
                    100,
                    context.getString(R.string.copying_builtin_models)
                )
                
                copiedFiles += copyAssetDirectoryRecursively(
                    context,
                    assetManager,
                    modelAssetPath,
                    modelLocalDir,
                    totalFiles,
                    copiedFiles,
                    onProgress
                )
            }
            
            onProgress?.invoke(100, 100, context.getString(R.string.copying_builtin_models))
            
            // Create symbolic links for compatibility with existing scanning logic
            for (modelDir in modelDirs) {
                val modelLocalDir = File(builtinModelsDir, modelDir)
                val symlinkFile = File(builtinModelsDir, modelDir)
                
                // The directory itself serves as the "symlink" for our scanning logic
                Log.d(TAG, "Model directory ready: ${modelLocalDir.absolutePath}")
            }
            
            // Merge any split files
            mergeSplitFiles(builtinModelsDir, modelDirs)
            
            // Mark builtin models as copied
            PreferenceUtils.setBoolean(context, BUILTIN_MODELS_COPIED_KEY, true)
            Log.d(TAG, "Successfully copied all builtin models")
            
            return@withContext true
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to copy builtin models", e)
            onProgress?.invoke(0, 100, "Copy failed: ${e.message}")
            return@withContext false
        }
    }
    
    /**
     * Count files in asset directory recursively
     */
    private fun countFilesInAssetDirectory(assetManager: android.content.res.AssetManager, assetPath: String): Int {
        return try {
            val files = assetManager.list(assetPath) ?: return 0
            var count = 0
            for (file in files) {
                val fullPath = "$assetPath/$file"
                val subFiles = assetManager.list(fullPath)
                if (subFiles != null && subFiles.isNotEmpty()) {
                    // It's a directory
                    count += countFilesInAssetDirectory(assetManager, fullPath)
                } else {
                    // It's a file
                    count++
                }
            }
            count
        } catch (e: Exception) {
            Log.w(TAG, "Error counting files in $assetPath", e)
            0
        }
    }
    
    /**
     * Copy asset directory recursively to local storage
     */
    private fun copyAssetDirectoryRecursively(
        context: Context,
        assetManager: android.content.res.AssetManager,
        assetPath: String,
        localDir: File,
        totalFiles: Int,
        currentProgress: Int,
        onProgress: ((current: Int, total: Int, message: String) -> Unit)?
    ): Int {
        var copiedCount = 0
        
        try {
            if (!localDir.exists()) {
                localDir.mkdirs()
            }
            
            val files = assetManager.list(assetPath) ?: return 0
            
            for (file in files) {
                val assetFilePath = "$assetPath/$file"
                val localFile = File(localDir, file)
                
                val subFiles = assetManager.list(assetFilePath)
                if (subFiles != null && subFiles.isNotEmpty()) {
                    // It's a directory
                    copiedCount += copyAssetDirectoryRecursively(
                        context,
                        assetManager,
                        assetFilePath,
                        localFile,
                        totalFiles,
                        currentProgress + copiedCount,
                        onProgress
                    )
                } else {
                    // It's a file
                    copyAssetFile(assetManager, assetFilePath, localFile)
                    copiedCount++
                    
                    // Check if this is a large file that needs splitting
                    if (localFile.length() > FileSplitter.MAX_CHUNK_SIZE) {
                        Log.d(TAG, "Large file detected: ${localFile.name} (${localFile.length()} bytes), splitting...")
                        val splitInfo = FileSplitter.splitFile(localFile, localDir)
                        if (splitInfo != null) {
                            Log.d(TAG, "Successfully split ${localFile.name} into ${splitInfo.totalChunks} chunks")
                            // Delete the original large file to save space
                            localFile.delete()
                        }
                    }
                    
                    // Update progress
                    val progress = ((currentProgress + copiedCount) * 100) / totalFiles
                    onProgress?.invoke(progress, 100, context.getString(R.string.copying_builtin_models))
                }
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Error copying asset directory $assetPath", e)
        }
        
        return copiedCount
    }
    
    /**
     * Merge split files in model directories
     */
    private suspend fun mergeSplitFiles(builtinModelsDir: File, modelDirs: Array<String>) {
        for (modelDir in modelDirs) {
            val modelLocalDir = File(builtinModelsDir, modelDir)
            if (FileSplitter.needsMerging(modelLocalDir)) {
                Log.d(TAG, "Merging split files in ${modelLocalDir.absolutePath}")
                val success = FileSplitter.mergeAllSplitFiles(modelLocalDir)
                if (success) {
                    Log.d(TAG, "Successfully merged split files in ${modelLocalDir.absolutePath}")
                } else {
                    Log.w(TAG, "Failed to merge some split files in ${modelLocalDir.absolutePath}")
                }
            }
        }
    }
    
    /**
     * Copy single asset file to local storage
     */
    private fun copyAssetFile(assetManager: android.content.res.AssetManager, assetPath: String, localFile: File) {
        try {
            assetManager.open(assetPath).use { inputStream ->
                FileOutputStream(localFile).use { outputStream ->
                    inputStream.copyTo(outputStream)
                }
            }
            Log.d(TAG, "Copied file: $assetPath -> ${localFile.absolutePath}")
        } catch (e: IOException) {
            Log.e(TAG, "Failed to copy file: $assetPath", e)
            throw e
        }
    }
    
    /**
     * Reset the copied flag - useful for testing or forcing re-copy
     */
    fun resetCopiedFlag(context: Context) {
        PreferenceUtils.setBoolean(context, BUILTIN_MODELS_COPIED_KEY, false)
        Log.d(TAG, "Reset builtin models copied flag")
    }
    
    /**
     * Check if builtin models have been copied
     */
    fun isBuiltinModelsCopied(context: Context): Boolean {
        return PreferenceUtils.getBoolean(context, BUILTIN_MODELS_COPIED_KEY, false)
    }
}
