package com.alibaba.mnntalk.model

import android.content.Context
import com.alibaba.mls.api.download.DownloadInfo
import com.alibaba.mls.api.download.DownloadListener
import com.alibaba.mls.api.download.ModelDownloadManager
import java.io.Closeable
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

data class VoiceModelSpec(
    val label: String,
    val modelId: String,
    val estimatedBytes: Long
)

sealed interface ModelBundleState {
    data object Missing : ModelBundleState
    data class Downloading(
        val label: String,
        val progress: Float,
        val downloadedBytes: Long,
        val totalBytes: Long
    ) : ModelBundleState

    data class Ready(val paths: VoiceModelPaths) : ModelBundleState
    data class Error(val message: String) : ModelBundleState
}

class VoiceModelBundleManager(
    context: Context,
    configuredDirectory: File? = null
) : DownloadListener, Closeable {
    companion object {
        val LLM = VoiceModelSpec(
            label = "对话模型",
            modelId = "ModelScope/MNN/Qwen3-0.6B-MNN",
            estimatedBytes = 454_473_462L
        )
        val ASR = VoiceModelSpec(
            label = "中英双语识别",
            modelId = "ModelScope/MNN/sherpa-mnn-streaming-zipformer-bilingual-zh-en-2023-02-20",
            estimatedBytes = 295_334_711L
        )
        val TTS = VoiceModelSpec(
            label = "中英双语音色",
            modelId = "ModelScope/MNN/bert-vits2-MNN",
            estimatedBytes = 1_392_023_806L
        )

        val MODEL_SPECS = listOf(LLM, ASR, TTS)
    }

    private val manager = ModelDownloadManager.getInstance(context.applicationContext)
    private val developerDirectories = listOfNotNull(
        configuredDirectory,
        File("/data/local/tmp/MNN"),
        context.getExternalFilesDir(null)?.let { File(it, "MNN") }
    ).distinctBy { it.absolutePath }
    private val closed = AtomicBoolean(false)
    private val stateMutable = MutableStateFlow<ModelBundleState>(ModelBundleState.Missing)
    val state: StateFlow<ModelBundleState> = stateMutable.asStateFlow()

    init {
        manager.addListener(this)
        refresh()
    }

    fun refresh() {
        resolvePaths()?.let {
            stateMutable.value = ModelBundleState.Ready(it)
            return
        }
        if (stateMutable.value !is ModelBundleState.Downloading) {
            stateMutable.value = ModelBundleState.Missing
        }
    }

    fun download() {
        if (closed.get()) return
        resolvePaths()?.let {
            stateMutable.value = ModelBundleState.Ready(it)
            return
        }
        downloadNextMissing()
    }

    fun readyPaths(): VoiceModelPaths? = resolvePaths()

    private fun downloadNextMissing() {
        val next = MODEL_SPECS.firstOrNull { downloadedDirectory(it) == null }
        if (next == null) {
            resolvePaths()?.let {
                stateMutable.value = ModelBundleState.Ready(it)
            } ?: run {
                stateMutable.value = ModelBundleState.Error("模型已经下载，但目录结构不完整")
            }
            return
        }
        updateProgress(next, manager.getDownloadInfo(next.modelId))
        manager.startDownload(next.modelId)
    }

    private fun resolvePaths(): VoiceModelPaths? {
        developerDirectories.forEach { directory ->
            VoiceModelDirectoryResolver.resolve(directory)?.let { return it }
        }

        val llmDirectory = downloadedDirectory(LLM) ?: return null
        val asrDirectory = downloadedDirectory(ASR) ?: return null
        val ttsDirectory = downloadedDirectory(TTS) ?: return null
        return VoiceModelDirectoryResolver.resolveDirectories(
            llmDirectory = llmDirectory,
            asrDirectory = asrDirectory,
            ttsDirectory = ttsDirectory,
            source = VoiceModelSource.DOWNLOADER
        )
    }

    private fun downloadedDirectory(spec: VoiceModelSpec): File? {
        return manager.getDownloadedFile(spec.modelId)?.takeIf { it.isDirectory && it.exists() }
    }

    private fun updateProgress(current: VoiceModelSpec, info: DownloadInfo) {
        val completedBytes = MODEL_SPECS
            .takeWhile { it != current }
            .filter { downloadedDirectory(it) != null }
            .sumOf { it.estimatedBytes }
        val currentBytes = if (info.savedSize > 0L) {
            info.savedSize.coerceAtMost(current.estimatedBytes)
        } else {
            (current.estimatedBytes * info.progress.coerceIn(0.0, 1.0)).toLong()
        }
        val total = MODEL_SPECS.sumOf { it.estimatedBytes }
        val downloaded = (completedBytes + currentBytes).coerceAtMost(total)
        stateMutable.value = ModelBundleState.Downloading(
            label = current.label,
            progress = if (total > 0L) downloaded.toFloat() / total else 0f,
            downloadedBytes = downloaded,
            totalBytes = total
        )
    }

    override fun onDownloadStart(modelId: String) {
        MODEL_SPECS.firstOrNull { it.modelId == modelId }?.let {
            updateProgress(it, manager.getDownloadInfo(modelId))
        }
    }

    override fun onDownloadProgress(modelId: String, downloadInfo: DownloadInfo) {
        MODEL_SPECS.firstOrNull { it.modelId == modelId }?.let {
            updateProgress(it, downloadInfo)
        }
    }

    override fun onDownloadFinished(modelId: String, path: String) {
        if (MODEL_SPECS.none { it.modelId == modelId }) return
        downloadNextMissing()
    }

    override fun onDownloadFailed(modelId: String, e: Exception) {
        val spec = MODEL_SPECS.firstOrNull { it.modelId == modelId } ?: return
        stateMutable.value = ModelBundleState.Error("${spec.label}下载失败：${e.message ?: "未知错误"}")
    }

    override fun close() {
        if (closed.compareAndSet(false, true)) {
            manager.removeListener(this)
        }
    }
}
