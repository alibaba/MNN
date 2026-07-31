package com.alibaba.mnntalk.engine

enum class Speaker {
    USER,
    ASSISTANT
}

data class TranscriptLine(
    val id: Long,
    val speaker: Speaker,
    val text: String,
    val isFinal: Boolean
)

data class GenerationMetrics(
    val promptTokens: Long,
    val generatedTokens: Long,
    val prefillMicros: Long,
    val decodeMicros: Long
) {
    val firstStageTokensPerSecond: Float
        get() = if (prefillMicros > 0L) promptTokens * 1_000_000f / prefillMicros else 0f

    val decodeTokensPerSecond: Float
        get() = if (decodeMicros > 0L) generatedTokens * 1_000_000f / decodeMicros else 0f
}

sealed interface VoiceChatState {
    data object Idle : VoiceChatState
    data class Preparing(val detail: String) : VoiceChatState
    data object Ready : VoiceChatState
    data class Listening(val partialText: String = "") : VoiceChatState
    data class Thinking(val userText: String) : VoiceChatState
    data class Speaking(val assistantText: String) : VoiceChatState
    data class Error(val message: String) : VoiceChatState
}
