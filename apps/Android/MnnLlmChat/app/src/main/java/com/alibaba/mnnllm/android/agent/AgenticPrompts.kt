// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.agent

data class AgentToolSpec(
    val name: String,
    val description: String,
    val inputSchema: String
)

data class AgentLoopBudget(
    val maxPasses: Int,
    val maxToolCalls: Int,
    val maxBrowserCalls: Int,
    val maxPythonCalls: Int
)

object AgenticPrompts {
    val defaultLoopBudget = AgentLoopBudget(
        maxPasses = 6,
        maxToolCalls = 12,
        maxBrowserCalls = 8,
        maxPythonCalls = 6
    )

    val supportedTools = listOf(
        AgentToolSpec(
            name = "get_current_time",
            description = "Return the current local date, time, timezone, weekday, and epoch milliseconds.",
            inputSchema = """{"type":"get_current_time"}"""
        ),
        AgentToolSpec(
            name = "browser_url",
            description = "Open or read a URL through the in-app browser layer and return page text, title, url, and status.",
            inputSchema = """{"url":"https://example.com","goal":"what to verify on this page"}"""
        ),
        AgentToolSpec(
            name = "web_search",
            description = "Search the web and return candidate result titles, snippets, and URLs for follow-up browser_url calls.",
            inputSchema = """{"type":"web_search","query":"search query"}"""
        ),
        AgentToolSpec(
            name = "python_exec",
            description = "Run bounded Python 3.11 code in the app sandbox for calculation, data analysis, py_compile checks, script storage, and Excel read/write helpers. Available packages include numpy, pandas, and openpyxl.",
            inputSchema = """{"type":"python_exec","code":"print(1+1)","input":"optional text or JSON","timeout_ms":15000,"output_files":[]}"""
        )
    )

    const val outputContract: String = """
Return either a final user reply or a JSON tool request. Tool requests must be valid JSON:
{
  "reply": "",
  "memory_updates": [],
  "skill_updates": [],
  "system_calls": [
    {
      "type": "get_current_time | web_search | browser_url | python_exec",
      "query": "for web_search",
      "url": "for browser_url",
      "code": "for python_exec",
      "input": "optional input for python_exec",
      "timeout_ms": 15000,
      "output_files": ["workspace-relative paths of files this call creates, such as sample.xlsx"]
    }
  ]
}

Rules:
- Prefer verifying fresh, factual, or high-impact claims with web_search and browser_url.
- Use browser_url to inspect primary sources, not only search snippets.
- Use get_current_time when exact current time matters.
- Use python_exec for math, statistics, parsing, table-like data processing, py_compile checks, reusable scripts, and Excel work.
- In python_exec, available packages include numpy, pandas, openpyxl, and standard-library modules such as csv, json, statistics, and collections.
- In python_exec, available helpers include read_excel(path), write_excel(filename, sheets), save_script(name, source), load_script(name), list_scripts(), run_script(name), compile_script(name), emit(value), and set_result(value).
- Python code must directly perform the requested action in the submitted script; do not only define functions without calling them.
- Python relative file paths are resolved inside the app's writable agent workspace.
- For creating simple Excel files, prefer write_excel(filename, sheets) over pandas so files are saved inside the app workspace and start quickly.
- When python_exec creates files, fill output_files with every generated workspace-relative filename/path.
- Files created or modified by python_exec are automatically returned to the chat as attachments.
- Do not import network/process/filesystem control modules such as requests, socket, subprocess, os, or sys. Use browser_url/web_search for network access and the provided workspace helpers for files.
- memory_updates are only for durable user preferences/facts. Do not store task status, generated file summaries, examples, apologies, or one-off results.
- skill_updates are only for reusable procedures with a short name, triggers, and action_template. Do not store one-off task descriptions.
- Keep memory/skill updates short: memory content <= 168 chars, skill action_template <= 350 chars.
- Do not expose raw tool JSON to the user in the final answer.
- If a tool fails, try another route when useful, then explain only the user-relevant limitation.
- Stop requesting tools when the current evidence is enough or the app reports that the loop budget is exhausted.
"""

    fun buildIdentityMemory(modelName: String, modelId: String): String {
        return buildString {
            appendLine("Host: MNN Chat, local on-device MNN LLM runtime.")
            appendLine("Model: ${modelName.ifBlank { "unknown" }} (${modelId.ifBlank { "unknown" }}).")
            appendLine("Tools are executed by the host app after valid system_calls JSON.")
        }.trim()
    }

    fun buildSystemPrompt(memoryBlock: String = "", skillBlock: String = ""): String {
        return buildString {
            appendLine("You are MNN Chat's on-device assistant.")
            appendLine("Use host tools via valid system_calls JSON when they improve accuracy or perform requested actions.")
            appendLine("For current web facts, search, browse, or URL inspection, request web_search/browser_url instead of saying you lack internet access.")
            appendLine("Answer directly when tools are unnecessary.")
            appendLine()
            appendLine("Available tools:")
            supportedTools.forEach { tool ->
                appendLine("- ${tool.name}: ${tool.description}")
                appendLine("  schema: ${tool.inputSchema}")
            }
            if (memoryBlock.isNotBlank()) {
                appendLine()
                appendLine("Relevant memory:")
                appendLine(memoryBlock)
            }
            if (skillBlock.isNotBlank()) {
                appendLine()
                appendLine("Relevant skills:")
                appendLine(skillBlock)
            }
            appendLine()
            appendLine(outputContract.trim())
        }
    }

    fun buildSystemPromptForModel(
        modelName: String,
        modelId: String,
        memoryBlock: String = "",
        skillBlock: String = ""
    ): String {
        val identityBlock = buildIdentityMemory(modelName, modelId)
        val mergedMemory = listOf(identityBlock, memoryBlock)
            .filter { it.isNotBlank() }
            .joinToString(separator = "\n\n")
        return buildSystemPrompt(memoryBlock = mergedMemory, skillBlock = skillBlock)
    }
}
