package com.alibaba.mnnllm.android.chat.agent

import com.google.gson.JsonArray
import com.google.gson.JsonElement
import com.google.gson.JsonNull
import com.google.gson.JsonObject
import com.google.gson.JsonPrimitive
import com.google.gson.stream.JsonReader
import com.google.gson.stream.JsonToken
import java.io.StringReader
import java.util.Locale

enum class AgentTool(val wireName: String) {
    TIME("get_current_time"), CALCULATOR("calculator"), SEARCH("web_search"),
    READ_PAGE("read_webpage"), MEMORY("save_memory");

    companion object {
        fun fromName(name: String): AgentTool? = when (name.trim().lowercase(Locale.ROOT)) {
            "get_current_time", "current_time", "time" -> TIME
            "calculator", "calculate", "calc", "math", "compute" -> CALCULATOR
            "web_search", "search" -> SEARCH
            "read_webpage", "browser_url", "browse_url", "web_browse", "open_url", "read_url" -> READ_PAGE
            "save_memory", "remember", "memory_update", "add_memory" -> MEMORY
            else -> null
        }
    }
}

data class AgentCall(val type: String, val arguments: Map<String, JsonElement> = emptyMap()) {
    val tool: AgentTool? get() = AgentTool.fromName(type)
}

data class AgentMemoryUpdate(val content: String, val category: String = "general")
data class AgentResponse(
    val reply: String? = null,
    val calls: List<AgentCall> = emptyList(),
    val memoryUpdates: List<AgentMemoryUpdate> = emptyList(),
    val skillUpdates: List<AgentSkill> = emptyList(),
) {
    val isEmpty: Boolean get() = reply.isNullOrEmpty() && calls.isEmpty() &&
        memoryUpdates.isEmpty() && skillUpdates.isEmpty()
}

data class ValidatedAgentCall(val tool: AgentTool, val value: String = "")

object AgentCallValidator {
    fun validate(call: AgentCall): ValidatedAgentCall {
        val tool = requireNotNull(call.tool) { "Unknown tool" }
        val keys = when (tool) {
            AgentTool.TIME -> emptySet()
            AgentTool.CALCULATOR -> setOf("expression", "expr", "formula", "input", "query", "q")
            AgentTool.SEARCH -> setOf("query", "q", "keyword", "keywords", "search_query")
            AgentTool.READ_PAGE -> setOf("url", "link", "query")
            AgentTool.MEMORY -> setOf("content", "memory", "fact", "preference", "query")
        }
        require(call.arguments.keys.all { it in keys }) { "Unexpected parameter" }
        if (tool == AgentTool.TIME) return ValidatedAgentCall(tool)
        if (tool == AgentTool.SEARCH && call.arguments.isEmpty()) return ValidatedAgentCall(tool)
        require(call.arguments.size == 1) { "Specify exactly one string parameter" }
        val element = call.arguments.values.single()
        require(element.isJsonPrimitive && element.asJsonPrimitive.isString) { "Parameter must be a string" }
        val value = element.asString.trim()
        val limit = when (tool) {
            AgentTool.CALCULATOR -> AgentCalculator.MAX_LENGTH
            AgentTool.SEARCH -> 160
            AgentTool.READ_PAGE -> 2048
            AgentTool.MEMORY -> 1000
            else -> 0
        }
        require((value.isNotEmpty() || tool == AgentTool.SEARCH) && value.length <= limit) { "Empty or oversized parameter" }
        require(value.none { it.code < 32 && it != '\n' && it != '\t' }) { "Control character in parameter" }
        if (tool == AgentTool.SEARCH) require(!value.contains('\n') && !value.contains('\t')) { "Use search keywords" }
        if (tool == AgentTool.READ_PAGE) require(AgentInputPolicy.resultNumber(value) != null || value.none { it.isWhitespace() }) { "Invalid URL" }
        return ValidatedAgentCall(tool, value)
    }
}

object AgentOutputParser {
    const val MAX_OUTPUT_LENGTH = 65536
    const val MAX_PARSED_CALLS = 32
    private const val MAX_DEPTH = 24
    private val wrapperKeys = setOf("reply", "answer", "system_calls", "tool_calls", "memory_updates", "skill_updates")
    private val fences = Regex("```(?:json)?\\s*([\\s\\S]*?)```", RegexOption.IGNORE_CASE)
    private val ordinaryFences = Regex("```([^\\r\\n`]*)\\r?\\n[\\s\\S]*?(?:```|$)")
    private val functions = Regex("<function=([^>\\s]+)\\s*>([\\s\\S]*?)</function(?:\\s*>|(?=<)|$)")
    private val parameters = Regex("<parameter=([^>\\s]+)\\s*>([\\s\\S]*?)(?:</parameter>|(?=<parameter=)|$)")

    fun parse(text: String): AgentResponse? {
        if (text.length > MAX_OUTPUT_LENGTH) return null
        return parseText(stripThinking(text), 0)
    }

    private fun parseText(text: String, depth: Int): AgentResponse? {
        if (depth > MAX_DEPTH) return null
        val protocolText = withoutOrdinaryFences(text)
        parseToolCallTags(protocolText)?.let { return it }
        for (candidate in candidates(protocolText)) {
            var response = parseElement(strictJson(candidate) ?: continue, depth) ?: continue
            val reply = response.reply
            if (response.calls.isEmpty() && reply != null) {
                parseText(reply, depth + 1)?.let { response = it }
            }
            if (response.isEmpty) continue
            if (response.reply.isNullOrEmpty() && response.calls.isEmpty() && candidate != text) {
                val prose = text.replace(candidate, "").replace("```json", "").replace("```", "").trim()
                if (prose.isNotEmpty()) response = response.copy(reply = prose)
            }
            return response
        }
        return null
    }

    fun parseToolCallTags(text: String): AgentResponse? {
        if (text.length > MAX_OUTPUT_LENGTH) return null
        val protocolText = withoutOrdinaryFences(text)
        if (!protocolText.contains("<tool_call>") && !protocolText.contains("<function=")) return null
        val calls = mutableListOf<AgentCall>()
        val blocks = if (protocolText.contains("<tool_call>")) {
            protocolText.split("<tool_call>").drop(1).take(MAX_PARSED_CALLS).map { it.substringBefore("</tool_call>") }
        } else listOf(protocolText)
        for (block in blocks) {
            val function = functions.find(block)
            if (function != null) {
                val args = linkedMapOf<String, JsonElement>()
                for (parameter in parameters.findAll(function.groupValues[2]).take(16)) {
                    val key = parameter.groupValues[1].lowercase(Locale.ROOT)
                    if (key in args) args["__duplicate_parameter"] = JsonPrimitive(true)
                    args[key] = JsonPrimitive(parameter.groupValues[2].trim())
                }
                calls.add(AgentCall(function.groupValues[1], args))
            } else {
                val candidate = firstJSONObject(block) ?: continue
                val parsed = strictJson(candidate)?.let { parseElement(it, 0) }
                calls.addAll(parsed?.calls.orEmpty().take(MAX_PARSED_CALLS - calls.size))
            }
            if (calls.size >= MAX_PARSED_CALLS) break
        }
        return calls.takeIf { it.isNotEmpty() }?.let { AgentResponse(calls = it) }
    }

    fun stripThinking(text: String): String {
        var result = text
        while (true) {
            val start = result.indexOf("<think>")
            if (start < 0) break
            val end = result.indexOf("</think>", start + 7)
            result = if (end < 0) result.substring(0, start) else result.removeRange(start, end + 8)
        }
        val close = result.indexOf("</think>")
        if (close >= 0) result = result.substring(close + 8)
        return result.trim()
    }

    fun visibleAnswer(partial: String): String? = answerText(partial, streaming = true).ifEmpty { null }

    fun finalAnswerText(text: String): String = answerText(text, streaming = false)

    private fun answerText(raw: String, streaming: Boolean): String {
        if (raw.length > MAX_OUTPUT_LENGTH) return ""
        val text = stripThinking(raw)
        val boundary = protocolBoundary(text, streaming)
        return (if (boundary == null) text else text.take(boundary)).trim()
    }

    private fun withoutOrdinaryFences(text: String): String = ordinaryFences.replace(text) { match ->
        val language = match.groupValues[1].trim().lowercase(Locale.ROOT)
        if (language.isNotEmpty() && language != "json") " ".repeat(match.value.length) else match.value
    }

    private fun protocolBoundary(text: String, streaming: Boolean, detectFences: Boolean = true): Int? {
        var index = 0
        while (index < text.length) {
            if (detectFences && text.startsWith("```", index)) {
                val headerEnd = text.indexOf('\n', index + 3)
                if (headerEnd < 0) {
                    if (protocolBoundary(text.substring(index + 3), streaming = false, detectFences = false) != null) return index
                    return if (streaming) index else null
                }
                val language = text.substring(index + 3, headerEnd).trim().lowercase(Locale.ROOT)
                val close = text.indexOf("```", headerEnd + 1)
                val end = if (close < 0) text.length else close
                val body = text.substring(headerEnd + 1, end)
                if ((language.isEmpty() || language == "json") &&
                    (streaming && close < 0 && body.isBlank() || protocolBoundary(body, streaming && close < 0, false) != null)) return index
                index = if (close < 0) text.length else close + 3
                continue
            }
            if (detectFences && text[index] == '`') {
                if (streaming && text.substring(index).all { it == '`' }) return index
                val close = text.indexOf('`', index + 1)
                if (close < 0) return if (streaming) index else null
                if (protocolBoundary(text.substring(index + 1, close), streaming = false, detectFences = false) != null) return index
                index = close + 1
                continue
            }
            if (text[index] == '<') {
                val tail = text.substring(index)
                if (tail.startsWith("<tool_call") || tail.startsWith("<function=") ||
                    streaming && listOf("<tool_call>", "<function=", "<think>", "</think>").any { it.startsWith(tail) }) return index
            }
            if (text[index] == '{' || text[index] == '[') {
                val end = jsonEnd(text, index)
                if (end == null) {
                    if (looksLikeJson(text.substring(index))) return index
                    index++
                    continue
                }
                val candidate = text.substring(index, end)
                val element = strictJson(candidate)
                if (element == null && looksLikeJson(candidate) || element != null && isProtocolElement(element)) return index
                index = end
                continue
            }
            index++
        }
        return null
    }

    private fun isProtocolElement(element: JsonElement, depth: Int = 0): Boolean {
        if (depth > MAX_DEPTH) return true
        if (element.isJsonObject && wrapperKeys.any { element.asJsonObject.has(it) }) return true
        if (element.isJsonArray && element.asJsonArray.any { isProtocolElement(it, depth + 1) }) return true
        return parseElement(element, depth)?.isEmpty == false
    }

    private fun looksLikeJson(text: String): Boolean {
        val tail = text.drop(1).trimStart()
        val next = tail.firstOrNull() ?: return true
        if (text.first() == '{') return next == '"' || next == '}'
        return next in "{[\"-0123456789]" || listOf("true", "false", "null").any {
            tail.startsWith(it) || it.startsWith(tail)
        }
    }

    private fun jsonEnd(text: String, start: Int): Int? {
        val stack = mutableListOf<Char>()
        var quoted = false
        var escaped = false
        for (index in start until text.length) {
            val ch = text[index]
            if (escaped) escaped = false
            else if (quoted && ch == '\\') escaped = true
            else if (ch == '"') quoted = !quoted
            else if (!quoted) {
                if (ch == '{' || ch == '[') stack.add(ch)
                if (stack.size > MAX_DEPTH) return null
                if (ch == '}' || ch == ']') {
                    if (stack.isEmpty() || stack.removeAt(stack.lastIndex) != if (ch == '}') '{' else '[') return null
                    if (stack.isEmpty()) return index + 1
                }
            }
        }
        return null
    }

    fun firstJSONObject(text: String): String? = balancedJson(text).firstOrNull { it.startsWith("{") }

    private fun candidates(text: String): List<String> =
        (listOf(text) + fences.findAll(text).take(8).map { it.groupValues[1].trim() } + balancedJson(text)).distinct()

    private fun balancedJson(text: String): List<String> {
        val result = mutableListOf<String>()
        var start = -1
        val stack = mutableListOf<Char>()
        var quoted = false
        var escaped = false
        for ((index, ch) in text.withIndex()) {
            if (start < 0) {
                if (ch != '{' && ch != '[') continue
                start = index
                stack.add(ch)
                continue
            }
            if (escaped) escaped = false
            else if (ch == '\\' && quoted) escaped = true
            else if (ch == '"') quoted = !quoted
            else if (!quoted) {
                if (ch == '{' || ch == '[') stack.add(ch)
                if (stack.size > MAX_DEPTH) return result
                if (ch == '}' || ch == ']') {
                    if (stack.isEmpty() || stack.removeAt(stack.lastIndex) != if (ch == '}') '{' else '[') return result
                    if (stack.isEmpty()) {
                        result.add(text.substring(start, index + 1))
                        start = -1
                        if (result.size == 8) break
                    }
                }
            }
        }
        return result
    }

    private fun strictJson(text: String): JsonElement? = try {
        JsonReader(StringReader(text)).use { reader ->
            reader.isLenient = false
            var nodes = 0
            fun readNode(depth: Int): JsonElement {
                require(depth <= MAX_DEPTH && ++nodes <= 4096)
                return when (reader.peek()) {
                    JsonToken.BEGIN_OBJECT -> {
                        reader.beginObject()
                        val objectValue = JsonObject()
                        while (reader.hasNext()) {
                            val name = reader.nextName()
                            require(!objectValue.has(name))
                            objectValue.add(name, readNode(depth + 1))
                        }
                        reader.endObject()
                        objectValue
                    }
                    JsonToken.BEGIN_ARRAY -> {
                        reader.beginArray()
                        val array = JsonArray()
                        while (reader.hasNext()) array.add(readNode(depth + 1))
                        reader.endArray()
                        array
                    }
                    JsonToken.STRING -> JsonPrimitive(reader.nextString())
                    JsonToken.NUMBER -> JsonPrimitive(reader.nextString().toBigDecimal())
                    JsonToken.BOOLEAN -> JsonPrimitive(reader.nextBoolean())
                    JsonToken.NULL -> { reader.nextNull(); JsonNull.INSTANCE }
                    else -> throw IllegalArgumentException("Invalid JSON")
                }
            }
            val element = readNode(0)
            require(reader.peek() == JsonToken.END_DOCUMENT)
            element
        }
    } catch (_: Exception) { null }

    private fun parseElement(element: JsonElement, depth: Int): AgentResponse? {
        if (depth > MAX_DEPTH) return null
        if (element.isJsonArray) return AgentResponse(calls = calls(element, depth + 1))
        if (!element.isJsonObject) return null
        val obj = element.asJsonObject
        if (wrapperKeys.none { obj.has(it) }) return singleCall(obj)?.let { AgentResponse(calls = listOf(it)) }
        val memories = obj.getAsJsonArrayOrNull("memory_updates")?.take(30)?.mapNotNull {
            if (string(it) != null) AgentMemoryUpdate(string(it)!!)
            else if (it.isJsonObject) {
                val memory = it.asJsonObject
                (string(memory["content"]) ?: string(memory["text"]))?.let { content ->
                    AgentMemoryUpdate(content, string(memory["category"]) ?: "general")
                }
            } else null
        }.orEmpty()
        val skills = obj.getAsJsonArrayOrNull("skill_updates")?.take(30)?.mapNotNull {
            if (!it.isJsonObject) return@mapNotNull null
            val skill = it.asJsonObject
            val name = string(skill["name"]) ?: return@mapNotNull null
            val action = string(skill["action_template"]) ?: string(skill["description"]) ?: return@mapNotNull null
            val triggers = skill["trigger_keywords"]
            val keywords = if (triggers?.isJsonArray == true) triggers.asJsonArray.take(8).mapNotNull(::string)
                else string(triggers)?.split(',', ';', '，')?.map { keyword -> keyword.trim() }.orEmpty()
            AgentSkill(name, keywords, action)
        }.orEmpty()
        return AgentResponse(string(obj["reply"]) ?: string(obj["answer"]),
            calls(obj["system_calls"] ?: obj["tool_calls"], depth + 1), memories, skills)
    }

    private fun calls(element: JsonElement?, depth: Int): List<AgentCall> {
        if (element == null || depth > MAX_DEPTH) return emptyList()
        return when {
            element.isJsonArray -> element.asJsonArray.take(MAX_PARSED_CALLS).flatMap { calls(it, depth + 1) }.take(MAX_PARSED_CALLS)
            element.isJsonObject -> singleCall(element.asJsonObject)?.let(::listOf).orEmpty()
            string(element) != null -> parseText(string(element)!!, depth + 1)?.calls.orEmpty()
            else -> emptyList()
        }
    }

    private fun singleCall(obj: JsonObject): AgentCall? {
        if (wrapperKeys.any { obj.has(it) }) return null
        val function = obj["function"]?.takeIf { it.isJsonObject }?.asJsonObject ?: obj
        val explicit = string(function["type"]) ?: string(function["name"]) ?: string(function["tool"])
        val type = explicit ?: when {
            function.has("url") -> "read_webpage"
            function.has("query") -> "web_search"
            else -> return null
        }
        val rawArgs = function["arguments"] ?: function["parameters"]
        val args = when {
            rawArgs == null -> function.entrySet().filter { it.key !in setOf("type", "name", "tool", "id") }
                .associate { it.key to it.value }
            rawArgs.isJsonObject -> rawArgs.asJsonObject.entrySet().associate { it.key to it.value }
            string(rawArgs) != null -> {
                val parsed = strictJson(string(rawArgs)!!)
                if (parsed?.isJsonObject == true) parsed.asJsonObject.entrySet().associate { it.key to it.value }
                else mapOf("__invalid_arguments" to rawArgs)
            }
            else -> mapOf("__invalid_arguments" to rawArgs)
        }
        return AgentCall(type, args)
    }

    private fun JsonObject.getAsJsonArrayOrNull(key: String): JsonArray? =
        get(key)?.takeIf { it.isJsonArray }?.asJsonArray

    private fun string(element: JsonElement?): String? = element?.takeIf {
        it.isJsonPrimitive && it.asJsonPrimitive.isString
    }?.asString?.trim()?.takeIf { it.isNotEmpty() }
}
