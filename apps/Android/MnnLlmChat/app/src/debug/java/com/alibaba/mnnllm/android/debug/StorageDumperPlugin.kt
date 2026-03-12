// Created by ruoyi.sjd on 2026/3/10.
// Copyright (c) 2026 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.debug

import com.alibaba.mnnllm.android.MnnLlmApplication
import com.alibaba.mnnllm.android.modelist.ModelDeletionHelper
import com.facebook.stetho.dumpapp.DumperContext
import com.facebook.stetho.dumpapp.DumperPlugin
import java.io.File
import java.io.PrintStream

/**
 * Stetho DumperPlugin for analyzing internal storage.
 *
 * Addresses issue #4233: Provides visibility into storage usage,
 * identifies orphan files, and allows cleanup of unused mmap caches.
 *
 * Usage via adb:
 *   dumpapp storage list                    - List all internal storage directories
 *   dumpapp storage analysis                - Get detailed storage analysis
 *   dumpapp storage mmap                    - List mmap cache directories
 *   dumpapp storage orphans                 - Find orphan mmap caches
 *   dumpapp storage clean                   - Clean orphan mmap caches
 *   dumpapp storage clean <path>            - Clean specific directory
 */
class StorageDumperPlugin : DumperPlugin {

    override fun getName(): String = "storage"

    override fun dump(dumpContext: DumperContext) {
        val writer = dumpContext.stdout
        val args = dumpContext.argsAsList

        if (args.isEmpty()) {
            doUsage(writer)
            return
        }

        when (args[0]) {
            "list" -> handleList(writer)
            "analysis" -> handleAnalysisWithArgs(writer, args)
            "mmap" -> handleMmap(writer)
            "orphans" -> handleOrphans(writer)
            "clean" -> handleClean(writer, args.drop(1))
            "verify" -> handleVerify(writer)
            else -> doUsage(writer)
        }
    }

    private fun handleList(writer: PrintStream) {
        val context = MnnLlmApplication.getAppContext()
        val filesDir = context.filesDir
        val dataDir = context.dataDir

        writer.println("=== Internal Storage Analysis ===")
        writer.println()

        writer.println("Data Directory: ${dataDir.absolutePath}")
        writer.println("Files Directory: ${filesDir.absolutePath}")
        writer.println()

        writer.println("=== Top-Level Directories ===")
        writer.println(String.format("%-50s %15s %10s", "Directory", "Size", "Files"))
        writer.println("-".repeat(75))

        // List data dir contents
        dataDir.listFiles()?.sortedByDescending { calculateDirSize(it) }?.forEach { file ->
            val size = if (file.isDirectory) calculateDirSize(file) else file.length()
            val fileCount = if (file.isDirectory) countFiles(file) else 1
            writer.println(String.format("%-50s %15s %10d",
                file.name,
                formatSize(size),
                fileCount
            ))
        }

        writer.println()
        writer.println("=== Files Directory Contents ===")
        writer.println(String.format("%-50s %15s %10s", "Directory", "Size", "Files"))
        writer.println("-".repeat(75))

        filesDir.listFiles()?.sortedByDescending { calculateDirSize(it) }?.forEach { file ->
            val size = if (file.isDirectory) calculateDirSize(file) else file.length()
            val fileCount = if (file.isDirectory) countFiles(file) else 1
            writer.println(String.format("%-50s %15s %10d",
                file.name,
                formatSize(size),
                fileCount
            ))
        }

        // Total summary
        writer.println()
        writer.println("=== Summary ===")
        val totalDataSize = calculateDirSize(dataDir)
        val totalFilesSize = calculateDirSize(filesDir)
        writer.println("Total data dir size: ${formatSize(totalDataSize)}")
        writer.println("Total files dir size: ${formatSize(totalFilesSize)}")
        writer.println("Free space: ${formatSize(filesDir.freeSpace)}")
        writer.println("Total space: ${formatSize(filesDir.totalSpace)}")
    }

    private fun handleAnalysis(writer: PrintStream) {
        val context = MnnLlmApplication.getAppContext()

        writer.println("=== Storage Analysis ===")
        writer.println()

        try {
            val analysis = ModelDeletionHelper.getStorageAnalysis(context)

            writer.println("Internal Storage:")
            writer.println("  Total: ${formatSize(analysis.internalStorageTotal)}")
            writer.println("  Used: ${formatSize(analysis.internalStorageUsed)}")
            writer.println("  Free: ${formatSize(analysis.internalStorageTotal - analysis.internalStorageUsed)}")
            writer.println()

            writer.println("Model Storage:")
            writer.println("  Model files: ${formatSize(analysis.modelStorageSize)}")
            writer.println("  Mmap cache: ${formatSize(analysis.totalMmapCacheSize)}")
            writer.println("  Orphan cache: ${formatSize(analysis.totalOrphanSize)}")
            writer.println()

            if (analysis.mmapCacheEntries.isNotEmpty()) {
                writer.println("Mmap Cache Entries:")
                writer.println(String.format("%-40s %15s %10s", "Model ID", "Size", "Status"))
                writer.println("-".repeat(65))

                analysis.mmapCacheEntries
                    .sortedByDescending { it.sizeBytes }
                    .forEach { entry ->
                        val status = if (entry.isOrphan) "ORPHAN" else "TRACKED"
                        writer.println(String.format("%-40s %15s %10s",
                            entry.modelId.take(40),
                            formatSize(entry.sizeBytes),
                            status
                        ))
                    }
            }

            if (analysis.totalOrphanSize > 0) {
                writer.println()
                writer.println("*** ${formatSize(analysis.totalOrphanSize)} can be freed by running: dumpapp storage clean ***")
            }

        } catch (e: Exception) {
            writer.println("Error during analysis: ${e.message}")
            e.printStackTrace(writer)
        }
    }

    private fun handleAnalysisWithArgs(writer: PrintStream, args: List<String>) {
        when {
            args.getOrNull(1) == "detail" -> handleAnalysisDetail(writer, entryModelId = null)
            args.size >= 2 -> handleAnalysisDetail(writer, entryModelId = args[1])
            else -> handleAnalysis(writer)
        }
    }

    private fun handleAnalysisDetail(writer: PrintStream, entryModelId: String?) {
        val context = MnnLlmApplication.getAppContext()
        try {
            val details = if (entryModelId != null) {
                val analysis = ModelDeletionHelper.getStorageAnalysis(context)
                val entry = analysis.mmapCacheEntries.find { it.modelId == entryModelId }
                if (entry == null) {
                    writer.println("No entry found for modelId: $entryModelId")
                    writer.println("Available entries: ${analysis.mmapCacheEntries.joinToString { it.modelId }}")
                    return
                }
                listOf(ModelDeletionHelper.getStorageDetailForEntry(context, entry))
            } else {
                ModelDeletionHelper.getStorageAnalysisWithDetails(context)
            }
            details.forEach { detail ->
                val status = if (detail.isOrphan) "ORPHAN" else "TRACKED"
                writer.println("=== Model: ${detail.entryModelId} ($status) ===")
                writer.println("Resolved modelId: ${detail.resolvedModelId ?: "—"}")
                writer.println("Model dir: ${detail.modelDirPath ?: "—"} (${formatSize(detail.modelDirSize)})")
                writer.println("Config dir: ${detail.configDirPath ?: "—"}")
                detail.configFiles.forEach { (path, size) -> writer.println("  $path  ${formatSize(size)}") }
                writer.println("Mmap dir: ${detail.mmapPath} (${formatSize(detail.mmapSize)})")
                detail.mmapFiles.forEach { (path, size) -> writer.println("  $path  ${formatSize(size)}") }
                writer.println()
            }
        } catch (e: Exception) {
            writer.println("Error during analysis detail: ${e.message}")
            e.printStackTrace(writer)
        }
    }

    private fun handleMmap(writer: PrintStream) {
        val context = MnnLlmApplication.getAppContext()
        val filesDir = context.filesDir

        writer.println("=== Mmap Cache Directories ===")
        writer.println()

        // List tmps directory (include nested tmps/<base>/modelers and tmps/<base>/modelscope per MmapUtils.getMmapDir)
        val tmpsDir = File(filesDir, "tmps")
        if (tmpsDir.exists() && tmpsDir.isDirectory) {
            writer.println("tmps/ (remote models mmap cache):")
            writer.println(String.format("  %-45s %15s %10s", "Directory", "Size", "Files"))
            writer.println("  " + "-".repeat(70))

            val tmpsEntries = mutableListOf<Pair<String, File>>()
            tmpsDir.listFiles()?.forEach { baseDir ->
                if (!baseDir.isDirectory) return@forEach
                var hasSourceSubdir = false
                for (subName in listOf("modelscope", "modelers")) {
                    val subDir = File(baseDir, subName)
                    if (subDir.exists() && subDir.isDirectory) {
                        hasSourceSubdir = true
                        tmpsEntries.add("${baseDir.name}/$subName" to subDir)
                    }
                }
                if (!hasSourceSubdir) {
                    tmpsEntries.add(baseDir.name to baseDir)
                }
            }
            tmpsEntries
                .sortedByDescending { (_, f) -> calculateDirSize(f) }
                .forEach { (name, dir) ->
                    writer.println(String.format("  %-45s %15s %10d",
                        name.take(45),
                        formatSize(calculateDirSize(dir)),
                        countFiles(dir)
                    ))
                }
            writer.println("  Total: ${formatSize(calculateDirSize(tmpsDir))}")
        } else {
            writer.println("tmps/ directory does not exist")
        }

        writer.println()

        // List local_temps directory
        val localTempsDir = File(filesDir, "local_temps")
        if (localTempsDir.exists() && localTempsDir.isDirectory) {
            writer.println("local_temps/ (local models mmap cache):")
            writer.println(String.format("  %-45s %15s %10s", "Directory", "Size", "Files"))
            writer.println("  " + "-".repeat(70))

            localTempsDir.listFiles()?.sortedByDescending { calculateDirSize(it) }?.forEach { dir ->
                if (dir.isDirectory) {
                    writer.println(String.format("  %-45s %15s %10d",
                        dir.name.take(45),
                        formatSize(calculateDirSize(dir)),
                        countFiles(dir)
                    ))
                }
            }
            writer.println("  Total: ${formatSize(calculateDirSize(localTempsDir))}")
        } else {
            writer.println("local_temps/ directory does not exist")
        }

        writer.println()

        // List builtin_temps directory
        val builtinTempsDir = File(filesDir, "builtin_temps")
        if (builtinTempsDir.exists() && builtinTempsDir.isDirectory) {
            writer.println("builtin_temps/ (builtin models mmap cache):")
            writer.println(String.format("  %-45s %15s %10s", "Directory", "Size", "Files"))
            writer.println("  " + "-".repeat(70))

            builtinTempsDir.listFiles()?.sortedByDescending { calculateDirSize(it) }?.forEach { dir ->
                if (dir.isDirectory) {
                    writer.println(String.format("  %-45s %15s %10d",
                        dir.name.take(45),
                        formatSize(calculateDirSize(dir)),
                        countFiles(dir)
                    ))
                }
            }
            writer.println("  Total: ${formatSize(calculateDirSize(builtinTempsDir))}")
        } else {
            writer.println("builtin_temps/ directory does not exist")
        }
    }

    private fun handleOrphans(writer: PrintStream) {
        val context = MnnLlmApplication.getAppContext()

        writer.println("=== Orphan Mmap Caches ===")
        writer.println()
        writer.println("Scanning for mmap cache directories not associated with any tracked model...")
        writer.println()

        try {
            val orphans = ModelDeletionHelper.findOrphanMmapCaches(context)

            if (orphans.isEmpty()) {
                writer.println("No orphan mmap caches found.")
                return
            }

            writer.println("Found ${orphans.size} orphan directories:")
            writer.println(String.format("%-50s %15s", "Path", "Size"))
            writer.println("-".repeat(65))

            var totalSize = 0L
            orphans.sortedByDescending { calculateDirSize(it) }.forEach { orphan ->
                val size = calculateDirSize(orphan)
                totalSize += size
                writer.println(String.format("%-50s %15s",
                    orphan.name.take(50),
                    formatSize(size)
                ))
            }

            writer.println("-".repeat(65))
            writer.println(String.format("%-50s %15s", "TOTAL", formatSize(totalSize)))
            writer.println()
            writer.println("Run 'dumpapp storage clean' to remove these orphan caches.")

        } catch (e: Exception) {
            writer.println("Error finding orphans: ${e.message}")
            e.printStackTrace(writer)
        }
    }

    private fun handleClean(writer: PrintStream, args: List<String>) {
        val context = MnnLlmApplication.getAppContext()

        if (args.isNotEmpty()) {
            // Clean specific directory
            val targetPath = args[0]
            val targetDir = File(context.filesDir, targetPath)

            if (!targetDir.exists()) {
                writer.println("Directory does not exist: $targetPath")
                return
            }

            if (!isInAllowedCleanupPath(context, targetDir)) {
                writer.println("Cannot clean directory outside allowed paths: $targetPath")
                writer.println("Allowed paths: tmps/, local_temps/, builtin_temps/")
                return
            }

            val size = calculateDirSize(targetDir)
            val fileCount = countFiles(targetDir)

            writer.println("Cleaning: ${targetDir.absolutePath}")
            writer.println("  Size: ${formatSize(size)}")
            writer.println("  Files: $fileCount")

            if (targetDir.deleteRecursively()) {
                writer.println("Successfully deleted.")
            } else {
                writer.println("Failed to delete directory.")
            }
            return
        }

        // Clean all orphans
        writer.println("=== Cleaning Orphan Mmap Caches ===")
        writer.println()

        try {
            val result = ModelDeletionHelper.cleanOrphanMmapCaches(context)

            if (result.success) {
                writer.println("Cleanup completed successfully!")
                writer.println("  Bytes freed: ${formatSize(result.bytesFreed)}")
                writer.println("  Files removed: ${result.filesRemoved}")
            } else {
                writer.println("Cleanup completed with errors:")
                writer.println("  Bytes freed: ${formatSize(result.bytesFreed)}")
                writer.println("  Files removed: ${result.filesRemoved}")
                writer.println("  Errors:")
                result.errors.forEach { error ->
                    writer.println("    - $error")
                }
            }

        } catch (e: Exception) {
            writer.println("Error during cleanup: ${e.message}")
            e.printStackTrace(writer)
        }
    }

    private fun isInAllowedCleanupPath(context: android.content.Context, file: File): Boolean {
        val filesDir = context.filesDir
        val allowedPaths = listOf(
            File(filesDir, "tmps"),
            File(filesDir, "local_temps"),
            File(filesDir, "builtin_temps")
        )
        return allowedPaths.any { allowed ->
            file.absolutePath.startsWith(allowed.absolutePath)
        }
    }

    private fun calculateDirSize(dir: File): Long {
        if (!dir.exists()) return 0
        return try {
            dir.walkTopDown()
                .filter { it.isFile }
                .map { it.length() }
                .sum()
        } catch (e: Exception) {
            0
        }
    }

    private fun countFiles(dir: File): Int {
        if (!dir.exists()) return 0
        return try {
            dir.walkTopDown()
                .filter { it.isFile }
                .count()
        } catch (e: Exception) {
            0
        }
    }

    private fun formatSize(bytes: Long): String {
        return when {
            bytes >= 1024L * 1024L * 1024L -> String.format("%.2f GB", bytes / (1024.0 * 1024.0 * 1024.0))
            bytes >= 1024L * 1024L -> String.format("%.2f MB", bytes / (1024.0 * 1024.0))
            bytes >= 1024L -> String.format("%.2f KB", bytes / 1024.0)
            else -> "$bytes B"
        }
    }

    /**
     * Machine-parseable verification of storage integrity.
     * Outputs raw byte values and verifies sum-of-parts == total.
     *
     * Output lines use key=value format for easy parsing by shell scripts.
     */
    private fun handleVerify(writer: PrintStream) {
        val context = MnnLlmApplication.getAppContext()
        val filesDir = context.filesDir
        val dataDir = context.dataDir
        var allPassed = true

        writer.println("=== Storage Verification ===")
        writer.println()

        // ---- 1. dataDir: sum of children == total ----
        val dataDirTotal = calculateDirSize(dataDir)
        val dataDirChildren = dataDir.listFiles() ?: emptyArray()
        val dataDirChildrenSum = dataDirChildren.sumOf { child ->
            if (child.isDirectory) calculateDirSize(child) else child.length()
        }

        writer.println("DATA_DIR_TOTAL_BYTES=$dataDirTotal")
        writer.println("DATA_DIR_CHILDREN_SUM_BYTES=$dataDirChildrenSum")
        writer.println("DATA_DIR_CHILDREN_COUNT=${dataDirChildren.size}")

        if (dataDirTotal == dataDirChildrenSum) {
            writer.println("CHECK_DATA_DIR_SUM=PASS")
        } else {
            writer.println("CHECK_DATA_DIR_SUM=FAIL (total=$dataDirTotal children_sum=$dataDirChildrenSum diff=${dataDirTotal - dataDirChildrenSum})")
            allPassed = false
        }
        writer.println()

        // ---- 2. filesDir: sum of children == total ----
        val filesDirTotal = calculateDirSize(filesDir)
        val filesDirChildren = filesDir.listFiles() ?: emptyArray()
        val filesDirChildrenSum = filesDirChildren.sumOf { child ->
            if (child.isDirectory) calculateDirSize(child) else child.length()
        }

        writer.println("FILES_DIR_TOTAL_BYTES=$filesDirTotal")
        writer.println("FILES_DIR_CHILDREN_SUM_BYTES=$filesDirChildrenSum")
        writer.println("FILES_DIR_CHILDREN_COUNT=${filesDirChildren.size}")

        if (filesDirTotal == filesDirChildrenSum) {
            writer.println("CHECK_FILES_DIR_SUM=PASS")
        } else {
            writer.println("CHECK_FILES_DIR_SUM=FAIL (total=$filesDirTotal children_sum=$filesDirChildrenSum diff=${filesDirTotal - filesDirChildrenSum})")
            allPassed = false
        }
        writer.println()

        // ---- 3. mmap cache: sum of entries == mmap total ----
        val tmpsDir = File(filesDir, "tmps")
        val localTempsDir = File(filesDir, "local_temps")
        val builtinTempsDir = File(filesDir, "builtin_temps")

        val tmpsSize = if (tmpsDir.exists()) calculateDirSize(tmpsDir) else 0L
        val localTempsSize = if (localTempsDir.exists()) calculateDirSize(localTempsDir) else 0L
        val builtinTempsSize = if (builtinTempsDir.exists()) calculateDirSize(builtinTempsDir) else 0L
        val mmapTotalCalculated = tmpsSize + localTempsSize + builtinTempsSize

        writer.println("MMAP_TMPS_BYTES=$tmpsSize")
        writer.println("MMAP_LOCAL_TEMPS_BYTES=$localTempsSize")
        writer.println("MMAP_BUILTIN_TEMPS_BYTES=$builtinTempsSize")
        writer.println("MMAP_TOTAL_BYTES=$mmapTotalCalculated")

        // Cross-check with ModelDeletionHelper analysis
        try {
            val analysis = ModelDeletionHelper.getStorageAnalysis(context)
            writer.println("ANALYSIS_MMAP_TOTAL_BYTES=${analysis.totalMmapCacheSize}")
            writer.println("ANALYSIS_ORPHAN_BYTES=${analysis.totalOrphanSize}")
            writer.println("ANALYSIS_MODEL_STORAGE_BYTES=${analysis.modelStorageSize}")

            if (mmapTotalCalculated == analysis.totalMmapCacheSize) {
                writer.println("CHECK_MMAP_TOTAL=PASS")
            } else {
                writer.println("CHECK_MMAP_TOTAL=FAIL (dir_scan=$mmapTotalCalculated analysis=${analysis.totalMmapCacheSize})")
                allPassed = false
            }

            // Verify orphan size <= total mmap size
            if (analysis.totalOrphanSize <= analysis.totalMmapCacheSize) {
                writer.println("CHECK_ORPHAN_LE_TOTAL=PASS")
            } else {
                writer.println("CHECK_ORPHAN_LE_TOTAL=FAIL (orphan=${analysis.totalOrphanSize} total=${analysis.totalMmapCacheSize})")
                allPassed = false
            }
        } catch (e: Exception) {
            writer.println("CHECK_MMAP_TOTAL=FAIL (exception: ${e.message})")
            writer.println("CHECK_ORPHAN_LE_TOTAL=FAIL (exception: ${e.message})")
            allPassed = false
        }
        writer.println()

        // ---- 4. Model storage directory ----
        val mnnModelsDir = File(filesDir, ".mnnmodels")
        val mnnModelsSize = if (mnnModelsDir.exists()) calculateDirSize(mnnModelsDir) else 0L
        writer.println("MODEL_STORAGE_BYTES=$mnnModelsSize")

        // ---- 5. filesDir is inside dataDir ----
        if (filesDirTotal <= dataDirTotal) {
            writer.println("CHECK_FILES_LE_DATA=PASS")
        } else {
            writer.println("CHECK_FILES_LE_DATA=FAIL (files=$filesDirTotal data=$dataDirTotal)")
            allPassed = false
        }
        writer.println()

        // ---- 6. Per-child detail for dataDir (for external cross-check) ----
        writer.println("--- dataDir children ---")
        dataDirChildren.sortedByDescending {
            if (it.isDirectory) calculateDirSize(it) else it.length()
        }.forEach { child ->
            val size = if (child.isDirectory) calculateDirSize(child) else child.length()
            writer.println("CHILD:${child.name}=$size")
        }
        writer.println()

        // ---- Final verdict ----
        if (allPassed) {
            writer.println("VERIFY_RESULT=ALL_PASS")
        } else {
            writer.println("VERIFY_RESULT=HAS_FAILURES")
        }
    }

    private fun doUsage(writer: PrintStream) {
        writer.println("Usage: dumpapp storage <command> [args]")
        writer.println()
        writer.println("Commands:")
        writer.println("  list                    - List all internal storage directories")
        writer.println("  analysis                - Get detailed storage analysis")
        writer.println("  analysis detail         - Drill-down: model dir, config dir, mmap files per entry")
        writer.println("  analysis <entryModelId> - Drill-down for a single entry (use list modelId)")
        writer.println("  mmap                    - List mmap cache directories")
        writer.println("  orphans                 - Find orphan mmap caches")
        writer.println("  clean                   - Clean all orphan mmap caches")
        writer.println("  clean <path>            - Clean specific directory (relative to filesDir)")
        writer.println("  verify                  - Machine-parseable storage integrity check")
        writer.println()
        writer.println("Examples:")
        writer.println("  dumpapp storage list")
        writer.println("  dumpapp storage analysis")
        writer.println("  dumpapp storage analysis detail")
        writer.println("  dumpapp storage analysis taobao-mnn_Qwen3.5-0.8B-MNN/modelers")
        writer.println("  dumpapp storage mmap")
        writer.println("  dumpapp storage orphans")
        writer.println("  dumpapp storage clean")
        writer.println("  dumpapp storage clean tmps/orphan-model")
        writer.println("  dumpapp storage verify")
    }
}
