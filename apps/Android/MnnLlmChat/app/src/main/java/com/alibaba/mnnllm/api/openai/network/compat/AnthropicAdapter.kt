package com.alibaba.mnnllm.api.openai.network.compat

import MessageContent
import MultiModalContent
import OpenAIChatRequest
import OpenAIGenericMessage
import OpenAIImageContent
import OpenAIMultiModalContent
import OpenAITextContent
import StandardImageUrl
import TextContent
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json

object AnthropicAdapter {
    private val json = Json { prettyPrint = false }

    fun toOpenAIRequest(request: AnthropicMessagesRequest): OpenAIChatRequest {
        val convertedMessages = mutableListOf<OpenAIGenericMessage>()

        if (!request.system.isNullOrBlank()) {
            convertedMessages.add(
                OpenAIGenericMessage(
                    role = "system",
                    content = TextContent(request.system)
                )
            )
        }

        request.messages.forEach { message ->
            convertedMessages.add(
                OpenAIGenericMessage(
                    role = message.role,
                    content = toOpenAIMessageContent(message)
                )
            )
        }

        return OpenAIChatRequest(
            model = request.model,
            messages = convertedMessages,
            temperature = request.temperature,
            top_p = request.top_p,
            stream = request.stream
        )
    }

    fun createNonStreamResponse(
        responseId: String,
        model: String,
        text: String,
        usage: AnthropicUsage
    ): String {
        val response = AnthropicMessagesResponse(
            id = responseId,
            model = model,
            content = listOf(AnthropicResponseContent(text = text)),
            usage = usage
        )
        return json.encodeToString(response)
    }

    private fun toOpenAIMessageContent(message: AnthropicMessage): MessageContent {
        if (message.content.isEmpty()) {
            return TextContent("")
        }

        val textBlocks = message.content.filter { it.type == "text" && !it.text.isNullOrBlank() }
        val nonTextBlocks = message.content.filter { it.type != "text" }
        if (textBlocks.size == 1 && nonTextBlocks.isEmpty()) {
            return TextContent(textBlocks.first().text ?: "")
        }

        val multimodalContents = mutableListOf<OpenAIMultiModalContent>()
        message.content.forEach { block ->
            when (block.type) {
                "text" -> {
                    if (!block.text.isNullOrBlank()) {
                        multimodalContents.add(OpenAITextContent(block.text))
                    }
                }

                "image" -> {
                    val imageUrl = block.source?.url
                    if (!imageUrl.isNullOrBlank()) {
                        multimodalContents.add(OpenAIImageContent(StandardImageUrl(url = imageUrl)))
                    }
                }
            }
        }

        return if (multimodalContents.isEmpty()) {
            TextContent("")
        } else {
            MultiModalContent(multimodalContents)
        }
    }
}
