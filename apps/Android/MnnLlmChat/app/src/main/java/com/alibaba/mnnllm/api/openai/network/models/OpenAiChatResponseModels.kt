package com.alibaba.mnnllm.api.openai.network.models

import kotlinx.serialization.Serializable

/**
 * OpenAI Chat API 响应数据模型
 * 负责定义聊天完成响应的数据结构
 */

@Serializable
data class DeltaResponse(
    val id: String,
    val `object`: String,
    val created: Long,
    val model: String,
    val choices: List<Choice>,
    val usage: Usage? = null // 仅在最后一个chunk中包含
)

@Serializable
data class Choice(
    val delta: Delta,
    val finish_reason : String? = null,
   // val index: Int  //后期要对传入请求检测index并在响应时添加index
)

@Serializable
data class Delta(
    val content: String
)

/**
 * Token使用统计信息
 * 符合OpenAI API标准
 */
@Serializable
data class Usage(
    val prompt_tokens: Int,
    val completion_tokens: Int,
    val total_tokens: Int
)

/**
 * 非流式响应数据模型
 */
@Serializable
data class ChatCompletionResponse(
    val id: String,
    val `object`: String,
    val created: Long,
    val model: String,
    val choices: List<CompletionChoice>,
    val usage: Usage
)

@Serializable
data class CompletionChoice(
    val message: Message,
    val finish_reason: String,
    val index: Int
)

@Serializable
data class Message(
    val role: String,
    val content: String
)