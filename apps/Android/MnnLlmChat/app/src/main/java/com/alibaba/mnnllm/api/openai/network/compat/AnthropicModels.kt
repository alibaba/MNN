package com.alibaba.mnnllm.api.openai.network.compat

import kotlinx.serialization.Serializable

@Serializable
data class AnthropicMessagesRequest(
    val model: String? = null,
    val messages: List<AnthropicMessage> = emptyList(),
    val system: String? = null,
    val stream: Boolean? = null,
    val temperature: Double? = null,
    val top_p: Double? = null,
    val max_tokens: Int? = null
)

@Serializable
data class AnthropicMessage(
    val role: String,
    val content: List<AnthropicContentBlock> = emptyList()
)

@Serializable
data class AnthropicContentBlock(
    val type: String,
    val text: String? = null,
    val source: AnthropicImageSource? = null
)

@Serializable
data class AnthropicImageSource(
    val type: String = "url",
    val url: String? = null,
    val media_type: String? = null,
    val data: String? = null
)

@Serializable
data class AnthropicUsage(
    val input_tokens: Int,
    val output_tokens: Int
)

@Serializable
data class AnthropicResponseContent(
    val type: String = "text",
    val text: String
)

@Serializable
data class AnthropicMessagesResponse(
    val id: String,
    val type: String = "message",
    val role: String = "assistant",
    val content: List<AnthropicResponseContent>,
    val model: String,
    val stop_reason: String = "end_turn",
    val stop_sequence: String? = null,
    val usage: AnthropicUsage
)
