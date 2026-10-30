package com.alibaba.mnnllm.api.openai.network.compat

import OpenAIGenericMessage
import TextContent
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import java.time.Instant

object OllamaAdapter {
    data class Request(val model: String, val messages: List<OpenAIGenericMessage>, val stream: Boolean)

    fun parse(body: JsonObject, chat: Boolean): Request {
        val supported = if (chat) setOf("model", "messages", "stream") else setOf("model", "prompt", "system", "stream")
        fun isNoOp(key: String, value: kotlinx.serialization.json.JsonElement): Boolean = when (key) {
            "tools", "images", "tool_calls" -> value is JsonNull || value is JsonArray && value.isEmpty()
            "options" -> value is JsonNull || value is JsonObject && value.isEmpty()
            "raw" -> value is JsonNull || value is JsonPrimitive && !value.isString && value.booleanOrNull == false
            else -> false
        }
        val unsupported = body.filter { (key, value) -> key !in supported && !isNoOp(key, value) }.keys
        require(unsupported.isEmpty()) { "Unsupported fields: ${unsupported.sorted().joinToString()}" }
        fun string(obj: JsonObject, key: String): String {
            val value = obj[key] as? JsonPrimitive
            require(value != null && value.isString) { "$key must be a string" }
            return value.content
        }
        val model = string(body, "model")
        require(model.isNotBlank()) { "model must not be empty" }
        val streamValue = body["stream"]
        val stream = if (streamValue == null || streamValue is JsonNull) true else
            (streamValue as? JsonPrimitive)?.takeIf { !it.isString }?.booleanOrNull
                ?: throw IllegalArgumentException("stream must be a boolean")
        val messages = if (chat) {
            val items = body["messages"] as? JsonArray ?: throw IllegalArgumentException("messages must be an array")
            require(items.isNotEmpty()) { "messages must not be empty" }
            items.mapIndexed { index, item ->
                val message = item as? JsonObject ?: throw IllegalArgumentException("message must be an object")
                require(message.none { (key, value) -> key !in setOf("role", "content") && !isNoOp(key, value) }) {
                    "Unsupported message fields"
                }
                val role = string(message, "role")
                require(role == "user" || role == "assistant" || (role == "system" && index == 0)) {
                    "Only user, assistant, and an initial system message are supported"
                }
                OpenAIGenericMessage(role, TextContent(string(message, "content")))
            }
        } else {
            val system = body["system"]?.takeUnless { it is JsonNull }?.let { string(body, "system") }
            val prompt = string(body, "prompt")
            buildList {
                if (!system.isNullOrEmpty()) add(OpenAIGenericMessage("system", TextContent(system)))
                add(OpenAIGenericMessage("user", TextContent(prompt)))
            }
        }
        return Request(model, messages, stream)
    }

    fun checkGenerationResult(metrics: Map<String, Any>) {
        val error = metrics["error"] as? String
        check(error.isNullOrEmpty()) { error ?: "Generation failed" }
    }

    fun response(model: String, text: String, done: Boolean, chat: Boolean, promptTokens: Int = 0, outputTokens: Int = 0): String =
        buildJsonObject {
            put("model", model)
            put("created_at", Instant.now().toString())
            if (chat) put("message", buildJsonObject { put("role", "assistant"); put("content", text) })
            else put("response", text)
            put("done", done)
            if (done) {
                put("prompt_eval_count", promptTokens)
                put("eval_count", outputTokens)
            }
        }.toString()
}
