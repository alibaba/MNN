package com.alibaba.mnnllm.api.openai.network.models

import kotlinx.serialization.Serializable

/** * OpenAI Chat API responsedatamodel * responsible fordefinechatcompleteresponsedatastructure*/

@Serializable
data class DeltaResponse(
    val id: String,
    val `object`: String,
    val created: Long,
    val model: String,
    val choices: List<Choice>,
    val usage: Usage? = null //only inlastchunkcontaining
)

@Serializable
data class Choice(
    val delta: Delta,
    val finish_reason : String? = null,
   //val index: Int //will need tofor incomingrequest detectionindexand in responsewhenadd index
)

@Serializable
data class Delta(
    val content: String
)

/** * Tokenusestatisticsinfo * comply withOpenAI APIstandard*/
@Serializable
data class Usage(
    val prompt_tokens: Int,
    val completion_tokens: Int,
    val total_tokens: Int
)

/** * non-streamingresponsedatamodel*/
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

/**
 * Models API responsedatamodel*/
@Serializable
data class ModelsResponse(
    val `object`: String = "list",
    val data: List<ModelData>
)

@Serializable
data class ModelData(
    val id: String,
    val `object`: String = "model",
    val created: Long,
    val owned_by: String = "mnn",
    val permission: List<ModelPermission> = emptyList(),
    val root: String? = null,
    val parent: String? = null
)

@Serializable
data class ModelPermission(
    val id: String,
    val `object`: String = "model_permission",
    val created: Long,
    val allow_create_engine: Boolean = false,
    val allow_sampling: Boolean = true,
    val allow_logprobs: Boolean = true,
    val allow_search_indices: Boolean = true,
    val allow_view: Boolean = true,
    val allow_fine_tuning: Boolean = false,
    val organization: String = "*",
    val group: String? = null,
    val is_blocking: Boolean = false
)