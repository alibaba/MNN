// Created by ruoyi.sjd on 2026/3/10.
// Copyright (c) 2026 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.modelist

import android.content.Context
import com.alibaba.mls.api.download.ModelDownloadManager
import com.alibaba.mnnllm.android.chat.model.ChatDataManager
import com.alibaba.mnnllm.android.history.HistoryUtils
import com.alibaba.mnnllm.android.modelsettings.ModelConfig
import com.alibaba.mnnllm.android.utils.MmapUtils
import java.io.File

/** Subdirs under tmps/<base> used by MmapUtils.getMmapDir() for ModelScope/Modelers; keep in sync with MmapUtils. */
private val TMPS_SOURCE_SUBDIRS = listOf("modelscope", "modelers")

/**
 * Helper class for model deletion with proper cleanup.
 *
 * Addresses issue #4233: Model files which were collected when model deletion
 * didn't work are now occupying huge extra disk space.
 *
 * This helper ensures that:
 * 1. Mmap cache files are cleaned up when a model is deleted
 * 2. Chat sessions and their resource directories are cleaned up (DB + HistoryUtils)
 * 3. Orphan files can be identified and cleaned
 */
object ModelDeletionHelper {

    /**
     * Result of model deletion with cleanup.
     */
    data class DeletionResult(
        val modelDeleted: Boolean,
        val mmapCacheCleared: Boolean,
        val chatSessionsCleared: Boolean,
        val errors: List<String> = emptyList()
    )

    /**
     * Result of orphan cleanup operation.
     */
    data class CleanupResult(
        val success: Boolean,
        val bytesFreed: Long,
        val filesRemoved: Int,
        val errors: List<String> = emptyList()
    )

    /**
     * Storage analysis entry for a single mmap cache directory.
     */
    data class MmapCacheEntry(
        val modelId: String,
        val path: String,
        val sizeBytes: Long,
        val isOrphan: Boolean
    )

    /**
     * Storage analysis report.
     */
    data class StorageAnalysis(
        val totalMmapCacheSize: Long,
        val totalOrphanSize: Long,
        val mmapCacheEntries: List<MmapCacheEntry>,
        val modelStorageSize: Long,
        val internalStorageTotal: Long,
        val internalStorageUsed: Long
    )

    /** Max file list size for drill-down to avoid huge output. */
    private const val MAX_DRILLDOWN_FILES = 100

    /**
     * Drill-down detail for a single mmap cache entry: model dir, config dir, and mmap file list with sizes.
     */
    data class ModelStorageDetail(
        val entryModelId: String,
        val resolvedModelId: String?,
        val modelDirPath: String?,
        val modelDirSize: Long,
        val configDirPath: String?,
        val configFiles: List<Pair<String, Long>>,
        val mmapPath: String,
        val mmapSize: Long,
        val mmapFiles: List<Pair<String, Long>>,
        val isOrphan: Boolean
    )

    /**
     * Resolve tracked modelId whose getMmapDir equals the given mmap path.
     */
    private fun resolveModelIdForMmapPath(trackedModelIds: List<String>, mmapPath: String): String? {
        for (modelId in trackedModelIds) {
            if (MmapUtils.getMmapDir(modelId) == mmapPath) return modelId
        }
        return null
    }

    /**
     * List files under a directory with sizes (path relative to dir), capped at [MAX_DRILLDOWN_FILES].
     */
    private fun listFilesWithSizes(dir: File): List<Pair<String, Long>> {
        if (!dir.exists() || !dir.isDirectory) return emptyList()
        val result = mutableListOf<Pair<String, Long>>()
        val basePath = dir.absolutePath
        dir.walkTopDown().maxDepth(3).filter { it.isFile }.take(MAX_DRILLDOWN_FILES).forEach { file ->
            val relative = file.absolutePath.removePrefix(basePath).trimStart('/')
            result.add(relative to file.length())
        }
        return result
    }

    /**
     * Get drill-down detail for a single mmap cache entry: model dir, config dir, mmap path and file list with sizes.
     */
    fun getStorageDetailForEntry(context: Context, entry: MmapCacheEntry): ModelStorageDetail {
        val trackedModelIds = ChatDataManager.getInstance(context).getAllDownloadedModels().map { it.modelId }
        val resolvedModelId = resolveModelIdForMmapPath(trackedModelIds, entry.path)

        var modelDirPath: String? = null
        var modelDirSize = 0L
        var configDirPath: String? = null
        var configFiles = emptyList<Pair<String, Long>>()

        if (resolvedModelId != null) {
            val modelDir = ModelDownloadManager.getInstance(context).getDownloadedFile(resolvedModelId)
            if (modelDir != null) {
                modelDirPath = modelDir.absolutePath
                modelDirSize = calculateDirSize(modelDir)
            }
            val configDir = File(ModelConfig.getModelConfigDir(resolvedModelId))
            if (configDir.exists()) {
                configDirPath = configDir.absolutePath
                configFiles = listFilesWithSizes(configDir)
            }
        }

        val mmapDir = File(entry.path)
        val mmapFiles = listFilesWithSizes(mmapDir)

        return ModelStorageDetail(
            entryModelId = entry.modelId,
            resolvedModelId = resolvedModelId,
            modelDirPath = modelDirPath,
            modelDirSize = modelDirSize,
            configDirPath = configDirPath,
            configFiles = configFiles,
            mmapPath = entry.path,
            mmapSize = entry.sizeBytes,
            mmapFiles = mmapFiles,
            isOrphan = entry.isOrphan
        )
    }

    /**
     * Get storage analysis with drill-down details for each mmap entry.
     */
    fun getStorageAnalysisWithDetails(context: Context): List<ModelStorageDetail> {
        val analysis = getStorageAnalysis(context)
        return analysis.mmapCacheEntries.map { getStorageDetailForEntry(context, it) }
    }

    /**
     * Delete a model with proper cleanup of mmap cache and chat sessions.
     *
     * @param context Android context
     * @param modelId The model ID to delete
     * @return DeletionResult with cleanup status
     */
    fun deleteModelWithCleanup(context: Context, modelId: String): DeletionResult {
        val errors = mutableListOf<String>()

        // 1. Clear mmap cache first (while model still exists for path calculation)
        val mmapCacheCleared = try {
            MmapUtils.clearMmapCache(modelId)
        } catch (e: Exception) {
            errors.add("Failed to clear mmap cache: ${e.message}")
            false
        }

        // 2. Clear chat sessions and their resource directories for this model
        val chatSessionsCleared = try {
            deleteSessionsByModelId(context, modelId)
            true
        } catch (e: Exception) {
            errors.add("Failed to clear chat sessions: ${e.message}")
            false
        }

        // 3. Delete the model files via ModelDownloadManager
        val modelDeleted = try {
            ModelDownloadManager.getInstance(context).deleteModel(modelId)
            true
        } catch (e: Exception) {
            errors.add("Failed to delete model: ${e.message}")
            false
        }

        return DeletionResult(
            modelDeleted = modelDeleted,
            mmapCacheCleared = mmapCacheCleared,
            chatSessionsCleared = chatSessionsCleared,
            errors = errors
        )
    }

    /**
     * Delete all chat sessions associated with a model (DB + session resource directories).
     */
    private fun deleteSessionsByModelId(context: Context, modelId: String) {
        val chatDataManager = ChatDataManager.getInstance(context)
        val sessions = chatDataManager.getSessionsForModel(modelId)
        for (session in sessions) {
            HistoryUtils.deleteHistory(context, chatDataManager, session.sessionId)
        }
    }

    /**
     * Find orphan mmap cache directories that are not associated with any tracked model.
     *
     * @param context Android context
     * @return List of orphan directories
     */
    fun findOrphanMmapCaches(context: Context): List<File> {
        val orphans = mutableListOf<File>()
        val filesDir = context.filesDir

        // Get all tracked model IDs from download history
        val chatDataManager = ChatDataManager.getInstance(context)
        val trackedModels = chatDataManager.getAllDownloadedModels().map { it.modelId }.toSet()

        // Scan tmps directory (include nested tmps/<base>/modelers and tmps/<base>/modelscope per MmapUtils.getMmapDir)
        val tmpsDir = File(filesDir, "tmps")
        if (tmpsDir.exists() && tmpsDir.isDirectory) {
            tmpsDir.listFiles()?.forEach { baseDir ->
                if (!baseDir.isDirectory) return@forEach
                var hasSourceSubdir = false
                for (subName in TMPS_SOURCE_SUBDIRS) {
                    val subDir = File(baseDir, subName)
                    if (subDir.exists() && subDir.isDirectory) {
                        hasSourceSubdir = true
                        val relativePath = "${baseDir.name}/$subName"
                        if (!isTrackedMmapCache(relativePath, trackedModels)) {
                            orphans.add(subDir)
                        }
                    }
                }
                if (!hasSourceSubdir) {
                    if (!isTrackedMmapCache(baseDir.name, trackedModels)) {
                        orphans.add(baseDir)
                    }
                }
            }
        }

        // Scan local_temps directory
        val localTempsDir = File(filesDir, "local_temps")
        if (localTempsDir.exists() && localTempsDir.isDirectory) {
            localTempsDir.listFiles()?.forEach { dir ->
                if (dir.isDirectory && !isTrackedLocalModel(dir.name, trackedModels)) {
                    orphans.add(dir)
                }
            }
        }

        // builtin_temps: do not treat as orphan (app-bundled models, not in download history)

        return orphans
    }

    /**
     * Check if a mmap cache directory corresponds to a tracked model.
     * @param dirName either "taobao-mnn_XXX" (HuggingFace) or "taobao-mnn_XXX/modelers" or "taobao-mnn_XXX/modelscope" (per MmapUtils.getMmapDir)
     */
    private fun isTrackedMmapCache(dirName: String, trackedModels: Set<String>): Boolean {
        val baseName = dirName.substringBefore('/')
        val subName = if (dirName.contains("/")) dirName.substringAfter('/') else null
        return trackedModels.any { modelId ->
            val safeName = modelId.replace("/", "_")
                .replace("HuggingFace_", "")
                .replace("ModelScope_", "")
                .replace("Modelers_", "")
                .replace("MNN_", "taobao-mnn_")
            when {
                subName == "modelers" -> safeName == baseName && modelId.startsWith("Modelers/")
                subName == "modelscope" -> safeName == baseName && modelId.startsWith("ModelScope/")
                subName == null -> safeName == baseName || baseName.startsWith(safeName)
                else -> false
            }
        }
    }

    /**
     * Check if a local model cache corresponds to a tracked model.
     */
    private fun isTrackedLocalModel(dirName: String, trackedModels: Set<String>): Boolean {
        return trackedModels.any { modelId ->
            modelId.startsWith("local/") && modelId.contains(dirName)
        }
    }

    /**
     * Clean all orphan mmap caches.
     *
     * @param context Android context
     * @return CleanupResult with statistics
     */
    fun cleanOrphanMmapCaches(context: Context): CleanupResult {
        val orphans = findOrphanMmapCaches(context)
        var bytesFreed = 0L
        var filesRemoved = 0
        val errors = mutableListOf<String>()

        for (orphan in orphans) {
            try {
                val size = calculateDirSize(orphan)
                val fileCount = countFiles(orphan)
                if (orphan.deleteRecursively()) {
                    bytesFreed += size
                    filesRemoved += fileCount
                } else {
                    errors.add("Failed to delete: ${orphan.absolutePath}")
                }
            } catch (e: Exception) {
                errors.add("Error deleting ${orphan.name}: ${e.message}")
            }
        }

        return CleanupResult(
            success = errors.isEmpty(),
            bytesFreed = bytesFreed,
            filesRemoved = filesRemoved,
            errors = errors
        )
    }

    /**
     * Get comprehensive storage analysis.
     *
     * @param context Android context
     * @return StorageAnalysis report
     */
    fun getStorageAnalysis(context: Context): StorageAnalysis {
        val filesDir = context.filesDir
        val chatDataManager = ChatDataManager.getInstance(context)
        val trackedModels = chatDataManager.getAllDownloadedModels().map { it.modelId }.toSet()

        val mmapCacheEntries = mutableListOf<MmapCacheEntry>()
        var totalMmapSize = 0L
        var totalOrphanSize = 0L

        // Analyze tmps directory (discover nested tmps/<base>/modelers and tmps/<base>/modelscope per MmapUtils.getMmapDir)
        val tmpsDir = File(filesDir, "tmps")
        if (tmpsDir.exists()) {
            tmpsDir.listFiles()?.forEach { baseDir ->
                if (!baseDir.isDirectory) return@forEach
                var hasSourceSubdir = false
                for (subName in TMPS_SOURCE_SUBDIRS) {
                    val subDir = File(baseDir, subName)
                    if (subDir.exists() && subDir.isDirectory) {
                        hasSourceSubdir = true
                        val size = calculateDirSize(subDir)
                        val relativePath = "${baseDir.name}/$subName"
                        val isOrphan = !isTrackedMmapCache(relativePath, trackedModels)
                        mmapCacheEntries.add(MmapCacheEntry(
                            modelId = relativePath,
                            path = subDir.absolutePath,
                            sizeBytes = size,
                            isOrphan = isOrphan
                        ))
                        totalMmapSize += size
                        if (isOrphan) totalOrphanSize += size
                    }
                }
                if (!hasSourceSubdir) {
                    val size = calculateDirSize(baseDir)
                    val isOrphan = !isTrackedMmapCache(baseDir.name, trackedModels)
                    mmapCacheEntries.add(MmapCacheEntry(
                        modelId = baseDir.name,
                        path = baseDir.absolutePath,
                        sizeBytes = size,
                        isOrphan = isOrphan
                    ))
                    totalMmapSize += size
                    if (isOrphan) totalOrphanSize += size
                }
            }
        }

        // Analyze local_temps directory
        val localTempsDir = File(filesDir, "local_temps")
        if (localTempsDir.exists()) {
            localTempsDir.listFiles()?.forEach { dir ->
                if (dir.isDirectory) {
                    val size = calculateDirSize(dir)
                    val isOrphan = !isTrackedLocalModel(dir.name, trackedModels)
                    mmapCacheEntries.add(MmapCacheEntry(
                        modelId = "local/${dir.name}",
                        path = dir.absolutePath,
                        sizeBytes = size,
                        isOrphan = isOrphan
                    ))
                    totalMmapSize += size
                    if (isOrphan) totalOrphanSize += size
                }
            }
        }

        // Analyze builtin_temps directory (builtin models; not considered orphan for cleanup)
        val builtinTempsDir = File(filesDir, "builtin_temps")
        if (builtinTempsDir.exists()) {
            builtinTempsDir.listFiles()?.forEach { dir ->
                if (dir.isDirectory) {
                    val size = calculateDirSize(dir)
                    mmapCacheEntries.add(MmapCacheEntry(
                        modelId = "builtin/${dir.name}",
                        path = dir.absolutePath,
                        sizeBytes = size,
                        isOrphan = false
                    ))
                    totalMmapSize += size
                }
            }
        }

        // Calculate model storage size
        val mnnModelsDir = File(filesDir, ".mnnmodels")
        val modelStorageSize = if (mnnModelsDir.exists()) calculateDirSize(mnnModelsDir) else 0L

        // Get internal storage stats
        val internalStorageTotal = filesDir.totalSpace
        val internalStorageUsed = internalStorageTotal - filesDir.freeSpace

        return StorageAnalysis(
            totalMmapCacheSize = totalMmapSize,
            totalOrphanSize = totalOrphanSize,
            mmapCacheEntries = mmapCacheEntries,
            modelStorageSize = modelStorageSize,
            internalStorageTotal = internalStorageTotal,
            internalStorageUsed = internalStorageUsed
        )
    }

    private fun calculateDirSize(dir: File): Long {
        if (!dir.exists()) return 0
        return dir.walkTopDown()
            .filter { it.isFile }
            .map { it.length() }
            .sum()
    }

    private fun countFiles(dir: File): Int {
        if (!dir.exists()) return 0
        return dir.walkTopDown()
            .filter { it.isFile }
            .count()
    }
}
