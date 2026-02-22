package com.alibaba.mnnllm.android.hotspot

import java.util.UUID

data class ChatUser(
    val id: String,
    val username: String,
    val language: String,       // ISO 639-1 code, e.g. "en", "ko", "ja"
    val avatarBase64: String?,
    val joinTime: Long = System.currentTimeMillis(),
)

data class ChatMessage(
    val id: String = UUID.randomUUID().toString(),
    val userId: String,
    val username: String,
    val text: String,
    val timestamp: Long = System.currentTimeMillis(),
    val replyToId: String? = null,
    val replyToText: String? = null,
    val replyToUsername: String? = null,
)

data class MessageTranslation(
    val messageId: String,
    val language: String,
    val text: String,
    val retranslationCount: Int = 0,
    val previousVersions: List<String> = emptyList(),
    val isPending: Boolean = false,
)

/**
 * A translation work item with a priority.
 * Lower number = higher priority:
 *   0 → UI text translation for a new language
 *   1 → newly sent message (FIFO within this tier)
 *   2 → retranslation with more context
 *   3 → historical messages for a newcomer (newest→oldest within this tier)
 */
sealed class TranslationTask(val priority: Int) : Comparable<TranslationTask> {
    abstract val key: String

    override fun compareTo(other: TranslationTask): Int =
        compareValuesBy(this, other, { it.priority })

    data class UiTranslationTask(
        val language: String,
        val requestId: String,
    ) : TranslationTask(0) {
        override val key get() = "ui:$language"
    }

    data class MessageTranslationTask(
        val messageId: String,
        val oldLanguage: String,
        val language: String,
        val contextCount: Int = 0,
        val previousTranslation: String? = null,
        val sequenceNumber: Long = System.currentTimeMillis(),
    ) : TranslationTask(if (contextCount > 0) 2 else 1) {
        override val key get() = "msg:$messageId:$language:$contextCount"
    }

    data class HistoryTranslationTask(
        val messageId: String,
        val oldLanguage: String,
        val language: String,
        val reverseOrder: Long,             // largest value = newest, so higher priority
    ) : TranslationTask(3) {
        override val key get() = "hist:$messageId:$language"

        override fun compareTo(other: TranslationTask): Int {
            if (other !is HistoryTranslationTask) return priority.compareTo(other.priority)
            // Within tier 3 we want newest first (largest reverseOrder first)
            return compareValuesBy(this, other, { it.priority }, { -it.reverseOrder })
        }
    }
}
