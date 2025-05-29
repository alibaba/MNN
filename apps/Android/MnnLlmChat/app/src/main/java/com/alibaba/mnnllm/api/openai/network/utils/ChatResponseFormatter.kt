package com.alibaba.mnnllm.api.openai.network.utils

import com.alibaba.mnnllm.api.openai.network.models.ChatCompletionResponse
import com.alibaba.mnnllm.api.openai.network.models.Choice
import com.alibaba.mnnllm.api.openai.network.models.CompletionChoice
import com.alibaba.mnnllm.api.openai.network.models.Delta
import com.alibaba.mnnllm.api.openai.network.models.DeltaResponse
import com.alibaba.mnnllm.api.openai.network.models.Message
import com.alibaba.mnnllm.api.openai.network.models.Usage
import kotlinx.serialization.json.Json

/**
 * 响应格式化工具
 * 负责创建和格式化各种响应数据
 */
class ChatResponseFormatter {

    private val json = Json { prettyPrint = false }

    /**
     * 创建Delta响应
     * @param responseId 响应ID
     * @param created 创建时间戳
     * @param content 内容
     * @param isLast 是否为最后一个chunk
     * @param usage 使用统计信息（仅在最后一个chunk中提供）
     * @return 格式化的JSON字符串
     */
    fun createDeltaResponse(
        responseId: String,
        created: Long,
        content: String,
        isLast: Boolean = false,
        usage: Usage? = null
    ): String {
        return json.encodeToString(
            DeltaResponse.serializer(), DeltaResponse(
                id = responseId,
                `object` = "chat.completion.chunk",
                created = created,
                model = "mnn-local",
                choices = listOf(
                    Choice(
                        delta = Delta(content),
                        finish_reason = if (isLast) "stop" else null,
                        // index = 0
                    )
                ),
                usage = if (isLast) usage else null // 仅在最后一个chunk中包含usage
            )
        )
    }

    /**
     * 创建非流式聊天完成响应
     * @param responseId 响应ID
     * @param created 创建时间戳
     * @param content 完整内容
     * @param usage 使用统计信息
     * @return 格式化的JSON字符串
     */
    fun createChatCompletionResponse(
        responseId: String,
        created: Long,
        content: String,
        usage: Usage
    ): String {
        return json.encodeToString(
            ChatCompletionResponse.serializer(), ChatCompletionResponse(
                id = responseId,
                `object` = "chat.completion",
                created = created,
                model = "mnn-local",
                choices = listOf(
                    CompletionChoice(
                        message = Message(
                            role = "assistant",
                            content = content
                        ),
                        finish_reason = "stop",
                        index = 0
                    )
                ),
                usage = usage
            )
        )
    }

    /**
     * 从LLM统计信息创建Usage对象
     * @param metrics LLM返回的统计信息HashMap
     * @return Usage对象
     */
    fun createUsageFromMetrics(metrics: HashMap<String, Any>): Usage {
        val promptTokens = (metrics["prompt_len"] as? Long)?.toInt() ?: 0
        val completionTokens = (metrics["decode_len"] as? Long)?.toInt() ?: 0
        return Usage(
            prompt_tokens = promptTokens,
            completion_tokens = completionTokens,
            total_tokens = promptTokens + completionTokens
        )
    }
}