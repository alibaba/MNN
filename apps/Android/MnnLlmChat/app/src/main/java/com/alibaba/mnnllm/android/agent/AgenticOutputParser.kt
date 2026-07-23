// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.agent

import com.google.gson.Gson
import com.google.gson.JsonArray
import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import com.google.gson.JsonSyntaxException

object AgenticOutputParser {
    private val gson = Gson()
    private val fencedJsonRegex = Regex("```(?:json)?\\s*([\\s\\S]*?)```", RegexOption.IGNORE_CASE)
    private val pythonExecLooseRegex = Regex(
        """"type"\s*:\s*"python_exec"[\s\S]*?"code"\s*:\s*"((?:\\.|[^"\\])*)"""",
        RegexOption.IGNORE_CASE
    )
    private val toolTypes = setOf(
        "get_current_time",
        "web_search",
        "browser_url",
        "python_exec",
        "run_python",
        "python"
    )

    fun parse(text: String): AgenticResponse? {
        for (candidate in candidates(text)) {
            val response = parseCandidate(candidate) ?: parseLooseToolCandidate(candidate) ?: continue
            if (response.systemCalls.isNullOrEmpty() && !response.reply.isNullOrBlank()) {
                val nested = parseNestedReply(response.reply)
                if (nested != null) {
                    return nested
                }
            }
            if (response.reply != null ||
                !response.systemCalls.isNullOrEmpty() ||
                !response.memoryUpdates.isNullOrEmpty() ||
                !response.skillUpdates.isNullOrEmpty()
            ) {
                return response
            }
        }
        return null
    }

    fun extractToolCalls(text: String): List<AgentSystemCall> {
        return parse(text)?.systemCalls.orEmpty()
    }

    private fun candidates(text: String): List<String> {
        val result = mutableListOf<String>()
        result.add(text.trim())
        fencedJsonRegex.findAll(text).forEach { match ->
            result.add(match.groupValues[1].trim())
        }
        extractFirstJsonObject(text)?.let { result.add(it) }
        return result.distinct().filter { it.isNotBlank() }
    }

    private fun parseCandidate(candidate: String): AgenticResponse? {
        return try {
            val element = JsonParser.parseString(candidate)
            when {
                element.isJsonArray -> parseToolArray(element.asJsonArray)
                element.isJsonObject -> {
                    val jsonObject = element.asJsonObject
                    parseSingleToolObject(jsonObject)
                        ?: parseResponseObject(jsonObject)
                }
                else -> null
            }
        } catch (_: JsonSyntaxException) {
            null
        } catch (_: IllegalStateException) {
            null
        }
    }

    private fun parseLooseToolCandidate(candidate: String): AgenticResponse? {
        val match = pythonExecLooseRegex.find(candidate) ?: return null
        val code = unescapeJsonString(match.groupValues[1])
        if (code.isBlank()) return null
        val input = Regex(""""input"\s*:\s*"((?:\\.|[^"\\])*)"""")
            .find(candidate)
            ?.groupValues
            ?.getOrNull(1)
            ?.let(::unescapeJsonString)
            .orEmpty()
        val timeoutMs = Regex(""""timeout_ms"\s*:\s*(\d+)""")
            .find(candidate)
            ?.groupValues
            ?.getOrNull(1)
            ?.toLongOrNull()
        val outputFiles = Regex(""""(?:output_files|generated_files|expected_outputs|files)"\s*:\s*\[([\s\S]*?)]""")
            .find(candidate)
            ?.groupValues
            ?.getOrNull(1)
            ?.let { raw ->
                Regex(""""((?:\\.|[^"\\])*)"""").findAll(raw)
                    .map { unescapeJsonString(it.groupValues[1]) }
                    .filter { it.isNotBlank() }
                    .toList()
            }
        return AgenticResponse(
            reply = "",
            systemCalls = listOf(
                AgentSystemCall(
                    type = "python_exec",
                    code = code,
                    input = input,
                    timeoutMs = timeoutMs,
                    outputFiles = outputFiles
                )
            )
        )
    }

    private fun unescapeJsonString(value: String): String {
        return runCatching {
            gson.fromJson("\"$value\"", String::class.java)
        }.getOrElse {
            value.replace("\\n", "\n")
                .replace("\\t", "\t")
                .replace("\\\"", "\"")
                .replace("\\\\", "\\")
        }
    }

    private fun parseResponseObject(jsonObject: JsonObject): AgenticResponse? {
        return AgenticResponse(
            reply = jsonObject.stringValue("reply"),
            memoryUpdates = parseMemoryUpdates(jsonObject.get("memory_updates")),
            skillUpdates = parseSkillUpdates(jsonObject.get("skill_updates")),
            systemCalls = parseSystemCalls(jsonObject.get("system_calls"))
        )
    }

    private fun parseSingleToolObject(jsonObject: JsonObject): AgenticResponse? {
        if (jsonObject.has("system_calls") ||
            jsonObject.has("reply") ||
            jsonObject.has("memory_updates") ||
            jsonObject.has("skill_updates")
        ) {
            return null
        }

        val explicitType = jsonObject.get("type")?.takeIf { it.isJsonPrimitive }?.asString?.trim()
        val inferredType = explicitType?.takeIf { it in toolTypes } ?: inferToolType(jsonObject)
        if (inferredType.isNullOrBlank()) {
            return null
        }

        val normalized = jsonObject.deepCopy()
        normalized.addProperty("type", inferredType)
        val call = gson.fromJson(normalized, AgentSystemCall::class.java)
        return AgenticResponse(reply = "", systemCalls = listOf(call))
    }

    private fun parseToolArray(jsonArray: JsonArray): AgenticResponse? {
        val calls = parseToolCallsFromArray(jsonArray)
        return calls.takeIf { it.isNotEmpty() }?.let {
            AgenticResponse(reply = "", systemCalls = it)
        }
    }

    private fun parseSystemCalls(element: JsonElement?): List<AgentSystemCall>? {
        if (element == null || element.isJsonNull) return null
        return when {
            element.isJsonArray -> parseToolCallsFromArray(element.asJsonArray)
            element.isJsonObject -> parseSingleToolObject(element.asJsonObject)?.systemCalls.orEmpty()
            element.isJsonPrimitive && element.asJsonPrimitive.isString ->
                parse(element.asString)?.systemCalls.orEmpty()
            else -> emptyList()
        }.takeIf { it.isNotEmpty() }
    }

    private fun parseToolCallsFromArray(jsonArray: JsonArray): List<AgentSystemCall> {
        return jsonArray.flatMap { element ->
            when {
                element.isJsonObject ->
                    parseSingleToolObject(element.asJsonObject)?.systemCalls.orEmpty()
                element.isJsonArray ->
                    parseToolCallsFromArray(element.asJsonArray)
                element.isJsonPrimitive && element.asJsonPrimitive.isString ->
                    parse(element.asString)?.systemCalls.orEmpty()
                else ->
                    emptyList()
            }
        }
    }

    private fun parseMemoryUpdates(element: JsonElement?): List<AgentMemoryUpdate>? {
        val array = element?.takeIf { it.isJsonArray }?.asJsonArray ?: return null
        return array.mapNotNull { item ->
            when {
                item.isJsonPrimitive && item.asJsonPrimitive.isString ->
                    AgentMemoryUpdate(category = "general", content = item.asString)
                item.isJsonObject -> {
                    val obj = item.asJsonObject
                    AgentMemoryUpdate(
                        category = obj.stringValue("category") ?: "general",
                        content = obj.stringValue("content") ?: obj.stringValue("text")
                    )
                }
                else -> null
            }
        }.takeIf { it.isNotEmpty() }
    }

    private fun parseSkillUpdates(element: JsonElement?): List<AgentSkillUpdate>? {
        val array = element?.takeIf { it.isJsonArray }?.asJsonArray ?: return null
        return array.mapNotNull { item ->
            when {
                item.isJsonPrimitive && item.asJsonPrimitive.isString ->
                    AgentSkillUpdate(description = item.asString)
                item.isJsonObject -> {
                    val obj = item.asJsonObject
                    AgentSkillUpdate(
                        name = obj.stringValue("name"),
                        description = obj.stringValue("description") ?: obj.stringValue("content"),
                        triggerKeywords = parseStringList(obj.get("trigger_keywords")),
                        actionTemplate = obj.stringValue("action_template")
                    )
                }
                else -> null
            }
        }.takeIf { it.isNotEmpty() }
    }

    private fun inferToolType(jsonObject: JsonObject): String? {
        return when {
            jsonObject.has("code") -> "python_exec"
            jsonObject.has("url") -> "browser_url"
            jsonObject.has("query") -> "web_search"
            else -> null
        }
    }

    private fun parseStringList(element: JsonElement?): List<String>? {
        if (element == null || element.isJsonNull) return null
        return when {
            element.isJsonArray -> element.asJsonArray.mapNotNull {
                it.takeIf { value -> value.isJsonPrimitive }?.asString
            }
            element.isJsonPrimitive && element.asJsonPrimitive.isString ->
                element.asString.split(',', ';').map { it.trim() }.filter { it.isNotBlank() }
            else -> null
        }
    }

    private fun JsonObject.stringValue(name: String): String? {
        val element = get(name) ?: return null
        if (!element.isJsonPrimitive || !element.asJsonPrimitive.isString) return null
        return element.asString.takeIf { it.isNotBlank() }
    }

    private fun parseNestedReply(reply: String): AgenticResponse? {
        for (candidate in candidates(reply)) {
            val nested = parseCandidate(candidate) ?: continue
            if (nested.reply != null ||
                !nested.systemCalls.isNullOrEmpty() ||
                !nested.memoryUpdates.isNullOrEmpty() ||
                !nested.skillUpdates.isNullOrEmpty()
            ) {
                return nested
            }
        }
        return null
    }

    private fun extractFirstJsonObject(text: String): String? {
        val start = text.indexOf('{')
        if (start < 0) return null

        var depth = 0
        var inString = false
        var escaping = false

        for (i in start until text.length) {
            val ch = text[i]
            if (escaping) {
                escaping = false
                continue
            }
            if (ch == '\\' && inString) {
                escaping = true
                continue
            }
            if (ch == '"') {
                inString = !inString
                continue
            }
            if (inString) continue

            when (ch) {
                '{' -> depth++
                '}' -> {
                    depth--
                    if (depth == 0) {
                        return text.substring(start, i + 1)
                    }
                }
            }
        }
        return null
    }
}
