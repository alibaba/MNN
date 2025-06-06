import kotlinx.serialization.InternalSerializationApi
import kotlinx.serialization.KSerializer
import kotlinx.serialization.Polymorphic
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.descriptors.PolymorphicKind
import kotlinx.serialization.descriptors.SerialDescriptor
import kotlinx.serialization.descriptors.buildSerialDescriptor
import kotlinx.serialization.descriptors.element
import kotlinx.serialization.encoding.Decoder
import kotlinx.serialization.encoding.Encoder
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonDecoder
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.decodeFromJsonElement

val json = Json {
    ignoreUnknownKeys = true
}

// ========================
// 请求结构
// ========================

@Serializable
data class OpenAIChatRequest(
    val model: String? = null,
    val messages: List<OpenAIGenericMessage>,
    val temperature: Double? = null,
    val top_p: Double? = null,
    val stream: Boolean? = null,
    val frequency_penalty: Double? = null,
    val presence_penalty: Double? = null,
)

// ========================
// 消息角色与内容
// ========================

@Serializable
@Polymorphic
sealed class OpenAIMessage {
    abstract val role: String?
}

@Serializable
data class OpenAIGenericMessage(
    override val role: String,
    @Serializable(with = MessageContentSerializer::class)
    val content: MessageContent?
) : OpenAIMessage()

// ========================
// 消息内容（单文本 or 多模态）
// ========================

@Serializable
data class OpenAITextContent(
    val text: String
) : OpenAIMultiModalContent() {
    override val type: String = "text"
}

@Serializable
@Polymorphic
sealed class MessageContent

@Serializable
data class TextContent(val text: String) : MessageContent()

@Serializable
data class MultiModalContent(val content: List<OpenAIMultiModalContent>) : MessageContent()

// ========================
// 多模态内容基类与实现
// ========================

@Serializable
@Polymorphic
sealed class OpenAIMultiModalContent {
    abstract val type: String
}

@Serializable
data class OpenAIImageContent(
    @SerialName("image_url")
    val imageUrl: StandardImageUrl
) : OpenAIMultiModalContent() {
    override val type: String = "image_url"
}

@Serializable
data class PreprocessedAudioContent(
    @SerialName("audio_url")
    val audioUrl: String
) : OpenAIMultiModalContent() {
    override val type: String = "audio_preprocess_required"
}

@Serializable
data class PreprocessedVideoContent(
    @SerialName("video_url")
    val videoUrl: String
) : OpenAIMultiModalContent() {
    override val type: String = "video_preprocess_required"
}

@Serializable
data class PreprocessedFileContent(
    @SerialName("file_url")
    val fileUrl: String,
    @SerialName("original_file_type")
    val originalFileType: String
) : OpenAIMultiModalContent() {
    override val type: String = "file_preprocess_required"
}

@Serializable
data class StandardImageUrl(
    val url: String,
    val detail: String? = null
)

// ========================
// 自定义消息内容序列化器
// ========================

object MessageContentSerializer : KSerializer<MessageContent> {

    @OptIn(InternalSerializationApi::class)
    override val descriptor: SerialDescriptor =
        buildSerialDescriptor("MessageContent", PolymorphicKind.SEALED) {
            element<TextContent>("text")
            element<MultiModalContent>("multi_modal")
        }

    override fun serialize(encoder: Encoder, value: MessageContent) {
        when (value) {
            is TextContent -> encoder.encodeSerializableValue(TextContent.serializer(), value)
            is MultiModalContent -> encoder.encodeSerializableValue(
                MultiModalContent.serializer(),
                value
            )
        }
    }

    override fun deserialize(decoder: Decoder): MessageContent {
        val input = decoder as JsonDecoder
        val jsonElement = input.decodeJsonElement()

        return when (jsonElement) {
            is JsonPrimitive -> TextContent(jsonElement.content)

            is JsonObject -> {
                val type = (jsonElement["type"] as? JsonPrimitive)?.content ?: "text"
                when (type) {
                    "text" -> json.decodeFromJsonElement<TextContent>(jsonElement)
                    "image_url" -> json.decodeFromJsonElement<OpenAIImageContent>(jsonElement)
                    "audio_preprocess_required" -> json.decodeFromJsonElement<PreprocessedAudioContent>(
                        jsonElement
                    )

                    "video_preprocess_required" -> json.decodeFromJsonElement<PreprocessedVideoContent>(
                        jsonElement
                    )

                    "file_preprocess_required" -> json.decodeFromJsonElement<PreprocessedFileContent>(
                        jsonElement
                    )

                    else -> throw IllegalArgumentException("Unknown content type: $type")
                }
            }

            is JsonArray -> {
                val contents = jsonElement.mapNotNull { item ->
                    when (item) {
                        is JsonObject -> {
                            val type = (item["type"] as? JsonPrimitive)?.content
                            when (type) {
                                "text" -> json.decodeFromJsonElement<OpenAITextContent>(item)
                                "image_url" -> json.decodeFromJsonElement<OpenAIImageContent>(item)
                                "audio_preprocess_required" -> json.decodeFromJsonElement<PreprocessedAudioContent>(
                                    item
                                )

                                "video_preprocess_required" -> json.decodeFromJsonElement<PreprocessedVideoContent>(
                                    item
                                )

                                "file_preprocess_required" -> json.decodeFromJsonElement<PreprocessedFileContent>(
                                    item
                                )

                                else -> null
                            }
                        }

                        is JsonPrimitive -> TextContent(item.content)
                        else -> null
                    }
                }.filterIsInstance<OpenAIMultiModalContent>()

                if (contents.size == 1 && contents.first() is TextContent) {
                    TextContent((contents.first() as TextContent).text)
                } else {
                    MultiModalContent(contents)
                }
            }

            else -> throw IllegalArgumentException("Unsupported content type: ${jsonElement::class}")
        } as MessageContent
    }
}