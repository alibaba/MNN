package com.alibaba.mnnllm.android.modelist

import android.content.Context
import com.alibaba.mls.api.ModelItem
import com.alibaba.mls.api.download.DownloadState
import com.alibaba.mls.api.download.ModelDownloadManager
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.utils.FileUtils
import java.io.File
import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.Date
import java.util.Locale

internal data class ModelListItemUiState(
    val title: String,
    val tags: List<String>,
    val statusText: String,
    val timeText: String? = null,
    val isPinned: Boolean = false,
    val updateButtonVisible: Boolean = false,
    val updateButtonText: String = "",
    val updateButtonEnabled: Boolean = true
)

internal object ModelListItemUiStateFactory {

    fun fromModelWrapper(
        context: Context,
        modelWrapper: ModelItemWrapper,
        modelDownloadManager: ModelDownloadManager = ModelDownloadManager.getInstance(context)
    ): ModelListItemUiState {
        val formattedSize = getFormattedFileSize(modelWrapper, modelDownloadManager)
        val downloadInfo = modelWrapper.modelItem.modelId?.let { modelDownloadManager.getDownloadInfo(it) }
        val isPreparing = downloadInfo?.downloadState == DownloadState.PREPARING
        val isUpdating = downloadInfo?.downloadState == DownloadState.DOWNLOADING || isPreparing

        val statusText: String
        val updateButtonVisible: Boolean
        val updateButtonText: String
        val updateButtonEnabled: Boolean

        if (modelWrapper.hasUpdate) {
            updateButtonVisible = true
            if (isUpdating) {
                updateButtonText = if (isPreparing) {
                    context.getString(R.string.download_pending)
                } else {
                    context.getString(R.string.download_state_updating)
                }
                updateButtonEnabled = false
                val statusMessage = if (isPreparing) {
                    context.getString(R.string.download_preparing)
                } else {
                    context.getString(R.string.download_state_updating)
                }
                statusText = if (formattedSize.isNotEmpty()) {
                    "$formattedSize ($statusMessage)"
                } else {
                    statusMessage
                }
            } else {
                updateButtonText = context.getString(R.string.update)
                updateButtonEnabled = true
                statusText = context.getString(R.string.downloaded_update_available, formattedSize)
            }
        } else {
            updateButtonVisible = false
            updateButtonText = ""
            updateButtonEnabled = true
            statusText = context.getString(R.string.downloaded_click_to_chat, formattedSize)
        }

        return ModelListItemUiState(
            title = modelWrapper.modelItem.modelName.orEmpty(),
            tags = getDisplayTagsWithSource(context, modelWrapper),
            statusText = statusText,
            timeText = formatTimeInfo(modelWrapper.lastChatTime),
            isPinned = modelWrapper.isPinned,
            updateButtonVisible = updateButtonVisible,
            updateButtonText = updateButtonText,
            updateButtonEnabled = updateButtonEnabled
        )
    }

    fun formatTimeInfo(
        lastChatTime: Long,
        now: Long = System.currentTimeMillis(),
        locale: Locale = Locale.getDefault()
    ): String? {
        if (lastChatTime <= 0) {
            return null
        }

        val chatDate = Date(lastChatTime)
        val today = Date(now)
        return if (isSameDay(chatDate, today)) {
            SimpleDateFormat("H:mm", locale).format(chatDate)
        } else {
            if (locale.language == "zh") {
                SimpleDateFormat("M月d日", locale).format(chatDate)
            } else {
                SimpleDateFormat("MMM d", locale).format(chatDate)
            }
        }
    }

    fun getFormattedFileSize(
        modelWrapper: ModelItemWrapper,
        modelDownloadManager: ModelDownloadManager
    ): String {
        val modelItem = modelWrapper.modelItem

        modelItem.modelId?.let { modelId ->
            val downloadedFile = modelDownloadManager.getDownloadedFile(modelId)
            if (downloadedFile != null) {
                return FileUtils.getFileSizeString(downloadedFile)
            }
        }

        if (modelWrapper.downloadSize > 0) {
            return FileUtils.formatFileSize(modelWrapper.downloadSize)
        }

        modelItem.localPath?.let { localPath ->
            val file = File(localPath)
            if (file.exists()) {
                return FileUtils.getFileSizeString(file)
            }
        }

        return ""
    }

    private fun isSameDay(date1: Date, date2: Date): Boolean {
        val cal1 = Calendar.getInstance()
        val cal2 = Calendar.getInstance()
        cal1.time = date1
        cal2.time = date2
        return cal1.get(Calendar.YEAR) == cal2.get(Calendar.YEAR) &&
            cal1.get(Calendar.DAY_OF_YEAR) == cal2.get(Calendar.DAY_OF_YEAR)
    }

    private fun getDisplayTags(modelItem: ModelItem, context: Context): List<String> {
        return com.alibaba.mnnllm.android.modelmarket.TagMapper
            .getDisplayTagList(modelItem.tags, context)
            .take(3)
    }

    private fun getDisplayTagsWithSource(
        context: Context,
        modelWrapper: ModelItemWrapper
    ): List<String> {
        val modelItem = modelWrapper.modelItem
        val contentTags = getDisplayTags(modelItem, context)
        val sourceLabel = modelWrapper.sourceTag
            ?: getModelSource(context, modelItem.modelId)
            ?: getModelSourceFromPath(context, modelItem.localPath)
        return if (sourceLabel != null) {
            listOf(sourceLabel) + contentTags.take(2)
        } else {
            contentTags.take(3)
        }
    }

    private fun getModelSource(context: Context, modelId: String?): String? {
        return when {
            modelId == null -> null
            modelId.startsWith("HuggingFace/") || modelId.contains("taobao-mnn") ->
                context.getString(R.string.huggingface)
            modelId.startsWith("ModelScope/") -> context.getString(R.string.modelscope)
            modelId.startsWith("Modelers/") -> context.getString(R.string.modelers)
            modelId.startsWith("Builtin/") -> context.getString(R.string.builtin)
            else -> null
        }
    }

    private fun getModelSourceFromPath(context: Context, localPath: String?): String? {
        if (localPath == null) return null
        return when {
            localPath.contains("/modelscope/") -> context.getString(R.string.modelscope)
            localPath.contains("/modelers/") -> context.getString(R.string.modelers)
            localPath.contains("/builtin/") -> context.getString(R.string.builtin)
            localPath.contains(".mnnmodels") -> context.getString(R.string.huggingface)
            else -> null
        }
    }
}
