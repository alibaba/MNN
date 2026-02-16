// Created by ruoyi.sjd on 2026/1/30.
// Copyright (c) 2026 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.debug

import com.alibaba.mls.api.download.DownloadInfo
import com.alibaba.mls.api.download.DownloadListener
import com.alibaba.mls.api.download.ModelDownloadManager
import com.alibaba.mls.api.source.ModelSources
import com.alibaba.mnnllm.android.MnnLlmApplication
import com.alibaba.mnnllm.android.mainsettings.MainSettings
import com.facebook.stetho.dumpapp.DumperContext
import com.facebook.stetho.dumpapp.DumperPlugin
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.io.PrintStream
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/**
 * Stetho DumperPlugin for debugging model downloads.
 * 
 * Usage via adb:
 *   dumpapp download source                     - Get current download source
 *   dumpapp download source set <SOURCE>        - Set download source (HuggingFace/ModelScope/Modelers)
 *   dumpapp download test <modelId>             - Test download a model
 *   dumpapp download status <modelId>           - Get download status of a model
 *   dumpapp download delete <modelId>           - Delete a downloaded model
 *   dumpapp download pause <modelId>            - Pause a download
 */
class DownloadDumperPlugin : DumperPlugin {

    override fun getName(): String {
        return "download"
    }

    override fun dump(dumpContext: DumperContext) {
        val writer = dumpContext.stdout
        val args = dumpContext.argsAsList

        if (args.isEmpty()) {
            doUsage(writer)
            return
        }

        val command = args[0]

        when (command) {
            "source" -> handleSource(writer, args.drop(1))
            "test" -> handleTest(writer, args.drop(1))
            "status" -> handleStatus(writer, args.drop(1))
            "delete" -> handleDelete(writer, args.drop(1))
            "pause" -> handlePause(writer, args.drop(1))
            else -> doUsage(writer)
        }
    }

    private fun handleSource(writer: PrintStream, args: List<String>) {
        val context = MnnLlmApplication.getAppContext()
        
        if (args.isEmpty()) {
            // Get current source
            val currentSource = MainSettings.getDownloadProviderString(context)
            val currentType = ModelSources.get().remoteSourceType
            writer.println("Current Download Source:")
            writer.println("  Provider String: $currentSource")
            writer.println("  Source Type: $currentType")
            writer.println()
            writer.println("Available sources: ${ModelSources.sourceList.joinToString(", ")}")
            return
        }

        if (args[0] == "set" && args.size >= 2) {
            val sourceName = args[1]
            val sourceType = when (sourceName.lowercase()) {
                "huggingface", "hf" -> {
                    MainSettings.setDownloadProvider(context, ModelSources.sourceHuffingFace)
                    ModelSources.ModelSourceType.HUGGING_FACE
                }
                "modelscope", "ms" -> {
                    MainSettings.setDownloadProvider(context, ModelSources.sourceModelScope)
                    ModelSources.ModelSourceType.MODEL_SCOPE
                }
                "modelers", "ml" -> {
                    MainSettings.setDownloadProvider(context, ModelSources.sourceModelers)
                    ModelSources.ModelSourceType.MODELERS
                }
                else -> {
                    writer.println("Unknown source: $sourceName")
                    writer.println("Available: HuggingFace (hf), ModelScope (ms), Modelers (ml)")
                    return
                }
            }
            ModelSources.setSourceType(sourceType)
            writer.println("Download source set to: $sourceName ($sourceType)")
        } else {
            writer.println("Usage: dumpapp download source set <SOURCE>")
            writer.println("Available sources: HuggingFace (hf), ModelScope (ms), Modelers (ml)")
        }
    }

    private fun handleTest(writer: PrintStream, args: List<String>) {
        if (args.isEmpty()) {
            writer.println("Usage: dumpapp download test <modelId>")
            writer.println("Examples:")
            writer.println("  dumpapp download test HuggingFace/taobao-mnn/Qwen2.5-0.5B-MNN")
            writer.println("  dumpapp download test ModelScope/MNN/Qwen2.5-0.5B-MNN")
            return
        }

        val modelId = args[0]
        writer.println("Starting download test for: $modelId")
        writer.println("Current source type: ${ModelSources.get().remoteSourceType}")
        writer.println()

        val context = MnnLlmApplication.getAppContext()
        val downloadManager = ModelDownloadManager.getInstance(context)
        
        // Create a latch for synchronous waiting (with timeout)
        val latch = CountDownLatch(1)
        var completed = false
        var resultMessage = ""

        val testListener = object : DownloadListener {
            override fun onDownloadStart(modelId: String) {
                writer.println("[START] Download started: $modelId")
            }

            override fun onDownloadProgress(modelId: String, downloadInfo: DownloadInfo) {
                val percent = (downloadInfo.progress * 100).toInt()
                val savedMB = downloadInfo.savedSize / (1024 * 1024)
                val totalMB = downloadInfo.totalSize / (1024 * 1024)
                val speed = downloadInfo.speedInfo ?: ""
                writer.println("[PROGRESS] $percent% - ${savedMB}MB/${totalMB}MB $speed")
            }

            override fun onDownloadFinished(modelId: String, absolutePath: String) {
                resultMessage = "[SUCCESS] Download finished: $absolutePath"
                completed = true
                latch.countDown()
            }

            override fun onDownloadFailed(modelId: String, e: Exception) {
                resultMessage = "[FAILED] Download failed: ${e.message}"
                e.printStackTrace(writer)
                latch.countDown()
            }

            override fun onDownloadPaused(modelId: String) {
                resultMessage = "[PAUSED] Download paused: $modelId"
                latch.countDown()
            }

            override fun onDownloadFileRemoved(modelId: String) {
                writer.println("[REMOVED] File removed: $modelId")
            }

            override fun onDownloadTotalSize(modelId: String, totalSize: Long) {
                val totalMB = totalSize / (1024 * 1024)
                writer.println("[INFO] Total size: ${totalMB}MB")
            }

            override fun onDownloadHasUpdate(modelId: String, downloadInfo: DownloadInfo) {
                writer.println("[UPDATE] Update available for: $modelId")
            }
        }

        downloadManager.addListener(testListener)
        
        try {
            downloadManager.startDownload(modelId)
            
            // Wait for result with timeout (5 minutes)
            val finished = latch.await(5, TimeUnit.MINUTES)
            
            if (!finished) {
                writer.println("[TIMEOUT] Download did not complete within 5 minutes")
            } else {
                writer.println(resultMessage)
            }
        } finally {
            downloadManager.removeListener(testListener)
        }
    }

    private fun handleStatus(writer: PrintStream, args: List<String>) {
        if (args.isEmpty()) {
            writer.println("Usage: dumpapp download status <modelId>")
            return
        }

        val modelId = args[0]
        val context = MnnLlmApplication.getAppContext()
        val downloadManager = ModelDownloadManager.getInstance(context)
        
        val info = downloadManager.getDownloadInfo(modelId)
        val state = downloadManager.getDownloadState(modelId)
        val downloadedFile = downloadManager.getDownloadedFile(modelId)

        writer.println("Download Status for: $modelId")
        writer.println("  State: ${info.downloadState}")
        writer.println("  Progress: ${(info.progress * 100).toInt()}%")
        writer.println("  Saved Size: ${info.savedSize / (1024 * 1024)}MB")
        writer.println("  Total Size: ${info.totalSize / (1024 * 1024)}MB")
        writer.println("  Speed: ${info.speedInfo ?: "N/A"}")
        writer.println("  Current File: ${info.currentFile ?: "N/A"}")
        writer.println("  Error: ${info.errorMessage ?: "None"}")
        writer.println("  Downloaded Path: ${downloadedFile?.absolutePath ?: "Not downloaded"}")
        writer.println("  Has Update: ${info.hasUpdate}")
    }

    private fun handleDelete(writer: PrintStream, args: List<String>) {
        if (args.isEmpty()) {
            writer.println("Usage: dumpapp download delete <modelId>")
            return
        }

        val modelId = args[0]
        val context = MnnLlmApplication.getAppContext()
        val downloadManager = ModelDownloadManager.getInstance(context)
        
        val before = downloadManager.getDownloadInfo(modelId)
        downloadManager.deleteModel(modelId)
        writer.println("Delete requested for: $modelId")
        writer.println("  Previous state: ${before.downloadState}")
    }

    private fun handlePause(writer: PrintStream, args: List<String>) {
        if (args.isEmpty()) {
            writer.println("Usage: dumpapp download pause <modelId>")
            return
        }

        val modelId = args[0]
        val context = MnnLlmApplication.getAppContext()
        val downloadManager = ModelDownloadManager.getInstance(context)
        
        downloadManager.pauseDownload(modelId)
        writer.println("Pause requested for: $modelId")
    }

    private fun doUsage(writer: PrintStream) {
        writer.println("Usage: dumpapp download <command> [args]")
        writer.println()
        writer.println("Commands:")
        writer.println("  source                      - Get current download source")
        writer.println("  source set <SOURCE>         - Set download source")
        writer.println("    SOURCE: HuggingFace (hf), ModelScope (ms), Modelers (ml)")
        writer.println()
        writer.println("  test <modelId>              - Test download a model")
        writer.println("  status <modelId>            - Get download status")
        writer.println("  delete <modelId>            - Delete downloaded model")
        writer.println("  pause <modelId>             - Pause active download")
        writer.println()
        writer.println("Examples:")
        writer.println("  dumpapp download source")
        writer.println("  dumpapp download source set hf")
        writer.println("  dumpapp download test HuggingFace/taobao-mnn/Qwen2.5-0.5B-MNN")
        writer.println("  dumpapp download status HuggingFace/taobao-mnn/Qwen2.5-0.5B-MNN")
    }
}
