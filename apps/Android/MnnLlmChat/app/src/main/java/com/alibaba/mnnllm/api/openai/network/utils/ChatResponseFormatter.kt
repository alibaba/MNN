package com.alibaba.mnnllm.api.openai.network.utils

import com.alibaba.mnnllm.api.openai.network.models.ChatCompletionResponse
import com.alibaba.mnnllm.api.openai.network.models.Choice
import com.alibaba.mnnllm.api.openai.network.models.CompletionChoice
import com.alibaba.mnnllm.api.openai.network.models.Delta
import com.alibaba.mnnllm.api.openai.network.models.DeltaResponse
import com.alibaba.mnnllm.api.openai.network.models.Message
import com.alibaba.mnnllm.api.openai.network.models.Usage
import kotlinx.serialization.json.Json

/** * responseformattool * responsible forcreateandformatvariousresponsedata*/
class ChatResponseFormatter {

    private val json = Json { prettyPrint = false }

    /** * createDeltaresponse * @param responseId responseID * @param created creation timestamp * @param content content * @param isLast whether aslastchunk * @param usage usage statisticsinfo（only inlastchunkprovide） * @return formatted JSONstring*/
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
                usage = if (isLast) usage else null //only inlastchunkcontainingusage
            )
        )
    }

    /** * createnon-streamingchatcompleteresponse * @param responseId responseID * @param created creation timestamp * @param content completecontent * @param usage usage statisticsinfo * @return formatted JSONstring*/
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

    /** * fromLLMstatisticsinfocreateUsageobject * @param metrics LLMreturnstatisticsinfoHashMap * @return Usageobject*/
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