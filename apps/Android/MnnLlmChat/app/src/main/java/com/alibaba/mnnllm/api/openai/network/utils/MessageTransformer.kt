package com.alibaba.mnnllm.api.openai.network.utils

import MultiModalContent
import OpenAIGenericMessage
import OpenAIImageContent
import OpenAITextContent
import PreprocessedAudioContent
import PreprocessedFileContent
import PreprocessedVideoContent
import TextContent
import android.util.Pair
import com.alibaba.mnnllm.android.llm.LlmSession
import com.alibaba.mnnllm.android.model.ModelUtils
import com.alibaba.mnnllm.api.openai.network.processors.MnnImageProcessor
import timber.log.Timber
import kotlin.collections.get

/**
 * 消息转换器
 * 负责将OpenAI格式的消息转换为MNN LLM所需的统一格式
 * 支持系统提示词合并、图片筛选等功能
 */
class MessageTransformer {

    companion object {
        private const val IMG_TAG_REGEX = "<img>.*?</img>"
    }

    /**
     * 将OpenAI格式的消息转换为MNN LLM所需的统一格式
     * @param messages OpenAI格式的消息列表
     * @param imageProcessor 图片处理器
     * @param llmSession LLM会话实例
     * @return 统一的历史消息列表（第一条为系统提示词，后续为角色对话）
     */
    suspend fun convertToUnifiedMnnHistory(
        messages: List<OpenAIGenericMessage>,
        imageProcessor: MnnImageProcessor,
        llmSession: LlmSession
    ): List<Pair<String, String>> {
        val unifiedHistory = mutableListOf<Pair<String, String>>()
        val isR1Model = ModelUtils.isR1Model(llmSession.modelId())

        val systemPromptPair = extractSystemPrompt(messages, imageProcessor, isR1Model)
        val systemPrompt = systemPromptPair.first
        val dialogMessages = systemPromptPair.second

        val finalSystemPrompt = buildFinalSystemPrompt(
            sessionPrompt = llmSession.getSystemPrompt(),
            rawSystemPrompt = systemPrompt,
            isR1Model = isR1Model
        )

        if (finalSystemPrompt.isNotEmpty()) {
            unifiedHistory.add(Pair("system", finalSystemPrompt))
        }

        addDialogMessagesToHistory(dialogMessages, imageProcessor, unifiedHistory, isR1Model)
        return unifiedHistory
    }

    // 提取系统提示词及剩余消息
    private suspend fun extractSystemPrompt(
        messages: List<OpenAIGenericMessage>,
        imageProcessor: MnnImageProcessor,
        isR1Model: Boolean
    ): Pair<String, List<OpenAIGenericMessage>> {
        var systemPrompt = ""
        var dialogMessages = messages

        if (messages.isNotEmpty() && messages.first().role == "system") {
            systemPrompt = processMessageContent(messages.first(), imageProcessor)
            if (isR1Model) {
                systemPrompt = "<|begin_of_sentence|>$systemPrompt"
            }
            dialogMessages = messages.drop(1)
        }

        return Pair(systemPrompt, dialogMessages)
    }

    // 构建最终的系统提示词
    private fun buildFinalSystemPrompt(
        sessionPrompt: String?,
        rawSystemPrompt: String,
        isR1Model: Boolean
    ): String {
        val systemPrompt = if (isR1Model) rawSystemPrompt else rawSystemPrompt
        return buildString {
            append(sessionPrompt.orEmpty())
            if (isNotEmpty() && systemPrompt.isNotEmpty()) append("\n")
            append(systemPrompt)
        }
    }

    // 处理并添加对话历史到结果中
    private suspend fun addDialogMessagesToHistory(
        messages: List<OpenAIGenericMessage>,
        imageProcessor: MnnImageProcessor,
        history: MutableList<Pair<String, String>>,
        isR1Model: Boolean
    ) {
        val lastImageIndex = findLastImageMessageIndex(messages)

        for ((index, message) in messages.withIndex()) {
            val role = when (message.role) {
                "assistant" -> "assistant"
                else -> "user"
            }

            var content = processMessageContent(message, imageProcessor)
            if (index != lastImageIndex) {
                content = content.replace(IMG_TAG_REGEX.toRegex(), "").trim()
            }

            if (isR1Model) {
                content = when (role) {
                    "user" -> "<|User|$content<|Assistant|>"
                    "assistant" -> "$content<|end_of_sentence|>"
                    else -> content
                }
            }

            history.add(Pair(role, content.trim()))
        }
    }

    /**
     * 处理单条消息的内容，支持文本和多模态内容
     */
    private suspend fun processMessageContent(
        message: OpenAIGenericMessage,
        imageProcessor: MnnImageProcessor
    ): String {
        return when (val content = message.content) {
            is String -> content
            is TextContent -> content.text
            is MultiModalContent -> processMultiModalContent(content, imageProcessor)
            is List<*> -> processLegacyListContent(content, imageProcessor)
            else -> ""
        } ?: ""
    }

    // 处理多模态内容（文本 + 图像 + 音频等）
    private suspend fun processMultiModalContent(
        content: MultiModalContent,
        imageProcessor: MnnImageProcessor
    ): String {
        var textPart = ""
        var imagePart = ""
        var audioPart = ""

        content.content.forEach { item ->
            when (item) {
                is OpenAITextContent -> {
                    textPart += if (textPart.isEmpty()) item.text else " ${item.text}"
                }

                is OpenAIImageContent -> {
                    try {
                        val imagePath = imageProcessor.processImageUrl(item.imageUrl.url)
                        if (imagePath != null) imagePart = "<img>$imagePath</img>"
                    } catch (e: Exception) {
                        Timber.Forest.tag("MessageTransformer").e(e, "Failed to process image: ${item.imageUrl.url}")
                    }
                }

                is PreprocessedAudioContent -> {
                    audioPart = "<audio>${item.audioUrl}</audio>"
                }

                is PreprocessedFileContent, is PreprocessedVideoContent -> Unit
            }
        }

        return when {
            imagePart.isNotEmpty() && textPart.isNotEmpty() -> "$imagePart$textPart"
            imagePart.isNotEmpty() -> imagePart
            audioPart.isNotEmpty() && textPart.isNotEmpty() -> "$audioPart$textPart"
            audioPart.isNotEmpty() -> audioPart
            else -> textPart
        }
    }

    // 兼容旧版 List<Map<String, Any?>> 格式处理
    private suspend fun processLegacyListContent(
        content: List<*>,
        imageProcessor: MnnImageProcessor
    ): String {
        return content.mapNotNull { item ->
            when (item) {
                is Map<*, *> -> {
                    when (item["type"]) {
                        "text" -> item["text"] as? String
                        "image_url" -> {
                            val url = (item["image_url"] as? Map<*, *>)?.get("url") as? String
                            url?.let { processImage(it, imageProcessor) }
                        }

                        else -> null
                    }
                }

                else -> null
            }
        }.joinToString("")
    }

    // 图片处理：下载/缓存/路径提取
    private suspend fun processImage(url: String, imageProcessor: MnnImageProcessor): String? {
        return try {
            val localPath = imageProcessor.processImageUrl(url)
            localPath?.let { "<img>$it</img>" }
        } catch (e: Exception) {
            Timber.Forest.tag("MessageTransformer").e(e, "Failed to process image: $url")
            null
        }
    }

    /**
     * 找到最后一条包含图片的消息索引
     * @param messages OpenAI格式的消息列表
     * @return 最后一条包含图片的消息索引，如果没有则返回-1
     */
    private fun findLastImageMessageIndex(messages: List<OpenAIGenericMessage>): Int {
        for (i in messages.indices.reversed()) {
            val content = messages[i].content
            when (content) {
                is MultiModalContent -> {
                    if (content.content.any { it is OpenAIImageContent }) return i
                }

                is List<*> -> {
                    if (content.any { item ->
                            (item as? Map<*, *>)?.get("type") == "image_url"
                        }) return i
                }

                is TextContent -> Unit
                null -> Unit
            }
        }
        return -1
    }
}