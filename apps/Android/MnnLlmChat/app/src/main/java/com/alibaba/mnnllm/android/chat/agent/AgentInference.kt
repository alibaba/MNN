package com.alibaba.mnnllm.android.chat.agent

data class AgentInferenceRequest(
    val messages: List<AgentMessage>,
    val finalOnly: Boolean,
    val allowThinking: Boolean,
    val outputTokenLimit: Int = 1024,
) {
    fun nativeHistory(): List<Pair<String, String>> = messages.map { it.role.wireName to it.content }
}

data class AgentGeneration(val text: String, val stopped: Boolean = false, val metrics: Map<String, Any> = emptyMap())
class AgentInferenceException(val metrics: Map<String, Any>) : IllegalStateException("Local inference failed")

interface AgentInference {
    suspend fun generate(request: AgentInferenceRequest, onDelta: (String) -> Unit): AgentGeneration
    suspend fun countTokens(messages: List<AgentMessage>): Int? = null
}

class AgentCallbackInference(
    private val generateCallback: suspend (AgentInferenceRequest, (String) -> Unit) -> AgentGeneration,
    private val tokenCounter: (suspend (List<AgentMessage>) -> Int)? = null,
) : AgentInference {
    override suspend fun generate(request: AgentInferenceRequest, onDelta: (String) -> Unit): AgentGeneration =
        generateCallback(request, onDelta)
    override suspend fun countTokens(messages: List<AgentMessage>): Int? = tokenCounter?.invoke(messages)
}

data class AgentStreamSnapshot(val answer: String, val thinking: String, val unclosedThinking: Boolean)

object AgentStreamDecoder {
    fun decode(raw: String, expectsThinking: Boolean): AgentStreamSnapshot {
        if (raw.contains("<|channel|>") || raw.contains("<|message|>")) {
            val final = raw.indexOf("final<|message|>")
            val thinking = raw.substringBefore("final<|message|>").substringAfter("<|message|>", "")
                .substringBefore("<|end|>").trim()
            val answer = if (final >= 0) raw.substring(final + "final<|message|>".length)
                .substringBefore("<|im_end|>").substringBefore("<|end|>").trim() else ""
            return AgentStreamSnapshot(answer, thinking, final < 0)
        }
        val implicitClose = raw.indexOf("</think>")
        if (expectsThinking && !raw.contains("<think>")) {
            return if (implicitClose < 0) AgentStreamSnapshot("", hidePartialTag(raw), raw.isNotBlank())
            else AgentStreamSnapshot(hidePartialTag(raw.substring(implicitClose + 8)).trim(), raw.substring(0, implicitClose).trim(), false)
        }
        val thinking = mutableListOf<String>()
        val answer = StringBuilder()
        var position = 0
        var unclosed = false
        while (position < raw.length) {
            val open = raw.indexOf("<think>", position)
            if (open < 0) {
                val remainder = raw.substring(position)
                if (remainder.contains("</think>")) {
                    thinking.add(remainder.substringBefore("</think>"))
                    answer.append(remainder.substringAfter("</think>"))
                } else answer.append(remainder)
                break
            }
            answer.append(raw.substring(position, open))
            val close = raw.indexOf("</think>", open + 7)
            if (close < 0) {
                thinking.add(hidePartialTag(raw.substring(open + 7)))
                unclosed = true
                break
            }
            thinking.add(raw.substring(open + 7, close))
            position = close + 8
        }
        return AgentStreamSnapshot(hidePartialTag(answer.toString()).substringBefore("<|im_end|>").trim(),
            thinking.joinToString("\n\n").trim(), unclosed)
    }

    private fun hidePartialTag(text: String): String {
        val index = text.lastIndexOf('<')
        if (index < 0) return text
        val tail = text.substring(index)
        return if (listOf("<think>", "</think>", "<tool_call>", "<function=", "<|im_end|>").any {
            it.startsWith(tail)
        }) text.substring(0, index) else text
    }
}
