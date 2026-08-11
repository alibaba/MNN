package com.alibaba.mnnllm.api.openai.network.compat

import kotlinx.serialization.KSerializer
import kotlinx.serialization.SerializationException
import kotlinx.serialization.Serializable
import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.descriptors.PrimitiveKind
import kotlinx.serialization.descriptors.PrimitiveSerialDescriptor
import kotlinx.serialization.descriptors.SerialDescriptor
import kotlinx.serialization.encoding.Decoder
import kotlinx.serialization.encoding.Encoder
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonDecoder
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive

@Serializable
data class AnthropicMessagesRequest(
    val model: String? = null,
    val messages: List<AnthropicMessage> = emptyList(),
    @Serializable(with = AnthropicSystemTextSerializer::class)
    val system: String? = null,
    val stream: Boolean? = null,
    val temperature: Double? = null,
    val top_p: Double? = null,
    val max_tokens: Int? = null
)

object AnthropicSystemTextSerializer : KSerializer<String?> {
    private val listSerializer = ListSerializer(AnthropicContentBlock.serializer())

    override val descriptor: SerialDescriptor = PrimitiveSerialDescriptor("AnthropicSystemText", PrimitiveKind.STRING)

    override fun serialize(encoder: Encoder, value: String?) {
        if (value == null) {
            encoder.encodeNull()
        } else {
            encoder.encodeString(value)
        }
    }

    override fun deserialize(decoder: Decoder): String? {
        if (decoder !is JsonDecoder) {
            return decoder.decodeString()
        }

        return when (val element = decoder.decodeJsonElement()) {
            JsonNull -> null
            is JsonPrimitive -> {
                if (element.isString) {
                    element.content
                } else {
                    throw SerializationException("Anthropic system primitive must be a string")
                }
            }
            is JsonObject -> decoder.json
                .decodeFromJsonElement(AnthropicContentBlock.serializer(), element)
                .text
            is JsonArray -> decoder.json
                .decodeFromJsonElement(listSerializer, element)
                .mapNotNull { it.text?.trim() }
                .filter { it.isNotEmpty() }
                .joinToString("\n")
        }
    }
}

@Serializable
data class AnthropicMessage(
    val role: String,
    @Serializable(with = AnthropicContentListSerializer::class)
    val content: List<AnthropicContentBlock> = emptyList()
)

object AnthropicContentListSerializer : KSerializer<List<AnthropicContentBlock>> {
    private val listSerializer = ListSerializer(AnthropicContentBlock.serializer())

    override val descriptor: SerialDescriptor = listSerializer.descriptor

    override fun serialize(encoder: Encoder, value: List<AnthropicContentBlock>) {
        encoder.encodeSerializableValue(listSerializer, value)
    }

    override fun deserialize(decoder: Decoder): List<AnthropicContentBlock> {
        if (decoder !is JsonDecoder) {
            return decoder.decodeSerializableValue(listSerializer)
        }

        return when (val element = decoder.decodeJsonElement()) {
            is JsonArray -> decoder.json.decodeFromJsonElement(listSerializer, element)
            is JsonObject -> listOf(decoder.json.decodeFromJsonElement(AnthropicContentBlock.serializer(), element))
            JsonNull -> emptyList()
            is JsonPrimitive -> {
                if (element.isString) {
                    listOf(AnthropicContentBlock(type = "text", text = element.content))
                } else {
                    throw SerializationException("Anthropic message content primitive must be a string")
                }
            }
        }
    }
}

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
