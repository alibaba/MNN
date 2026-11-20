package com.alibaba.mnnllm.android.history

import android.net.Uri
import com.alibaba.mnnllm.android.chat.model.ChatDataItem

internal object HistoryDisplayData {
    // Sanitize display copies only: missing media must never rewrite the saved conversation.
    fun prepare(
        items: List<ChatDataItem>,
        unavailableText: String,
        isAvailable: (Uri) -> Boolean
    ): MutableList<ChatDataItem> = items.map { source ->
        var missingAttachment = false
        fun available(uri: Uri?): Uri? {
            if (uri == null) return null
            if (isAvailable(uri)) return uri
            missingAttachment = true
            return null
        }
        ChatDataItem(source.time, source.type, source.text).apply {
            displayText = source.displayText
            thinkingText = source.thinkingText
            thinkingFinishedTime = source.thinkingFinishedTime
            showThinking = source.showThinking
            benchmarkInfo = source.benchmarkInfo
            imageUris = source.imageUris?.mapNotNull { available(it) }
            audioUri = available(source.audioUri)
            audioDuration = source.audioDuration
            videoUri = available(source.videoUri)
            if (missingAttachment) {
                text = listOfNotNull(text?.takeIf { it.isNotEmpty() }, unavailableText).joinToString("\n\n")
                displayText = listOfNotNull(displayText?.takeIf { it.isNotEmpty() }, unavailableText).joinToString("\n\n")
            }
        }
    }.toMutableList()
}
