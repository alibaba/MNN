// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.agent

import android.util.Base64
import android.util.Log
import com.alibaba.mnnllm.android.chat.model.ChatFileAttachment
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull
import java.io.BufferedReader
import java.io.File
import java.io.InputStream
import java.io.InputStreamReader
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLDecoder
import java.net.URLEncoder
import java.time.LocalDateTime
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import java.time.format.TextStyle
import java.util.Locale
import java.util.zip.GZIPInputStream

object AgenticToolExecutor {
    private const val TAG = "AgenticToolExecutor"
    private const val MAX_SEARCH_RESULTS = 5
    private const val MAX_PAGE_CHARS = 16_000
    private const val USER_AGENT =
        "Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Mobile Safari/537.36"
    private val fileReferenceRegex = Regex(
        """(?i)(?:file://)?/?[\p{L}\p{N}_ .\-/\\%]+?\.(?:xlsx|xls|csv|pdf|txt|json|png|jpe?g|py)"""
    )

    data class ToolStepEvent(
        val type: Type,
        val title: String,
        val detail: String = ""
    ) {
        enum class Type { STARTED, FINISHED, FAILED }
    }

    data class ToolExecutionResult(
        val observations: String,
        val generatedFiles: List<ChatFileAttachment> = emptyList()
    )

    private data class SearchHit(
        val title: String,
        val url: String,
        val snippet: String
    )

    suspend fun execute(
        calls: List<AgentSystemCall>,
        onStep: suspend (ToolStepEvent) -> Unit = {}
    ): ToolExecutionResult {
        val results = mutableListOf<String>()
        val generatedFiles = mutableListOf<ChatFileAttachment>()
        for (call in calls) {
            currentCoroutineContext().ensureActive()
            executeOne(call, onStep)?.let { result ->
                results.add(result.observation)
                generatedFiles.addAll(result.generatedFiles)
            }
        }
        return ToolExecutionResult(
            observations = results.joinToString("\n---\n"),
            generatedFiles = generatedFiles
        )
    }

    fun resolveMentionedWorkspaceFiles(text: String): List<ChatFileAttachment> {
        if (text.isBlank()) return emptyList()
        val references = fileReferenceRegex.findAll(text)
            .map { it.value }
            .toList()
        return resolveWorkspaceFileReferences(references)
    }

    fun resolveWorkspaceFileReferences(references: List<String>): List<ChatFileAttachment> {
        if (references.isEmpty()) return emptyList()
        return references.mapNotNull { resolveWorkspaceFileReference(it) }
            .distinctBy { it.path }
    }

    private data class SingleToolResult(
        val observation: String,
        val generatedFiles: List<ChatFileAttachment> = emptyList()
    )

    private suspend fun executeOne(
        call: AgentSystemCall,
        onStep: suspend (ToolStepEvent) -> Unit
    ): SingleToolResult? {
        val title = toolTitle(call)
        val detail = toolDetail(call)
        onStep(ToolStepEvent(ToolStepEvent.Type.STARTED, title, detail))
        return try {
            val result = when (normalizeType(call.type)) {
                "get_current_time" -> SingleToolResult(executeGetCurrentTime())
                "web_search" -> SingleToolResult(executeWebSearch(call.query.orEmpty()))
                "browser_url", "browse_url", "web_browse", "open_url" ->
                    SingleToolResult(executeBrowseUrl(call.url.orEmpty().ifBlank { call.query.orEmpty() }))
                "python_exec", "run_python", "python" -> executePython(call)
                else -> SingleToolResult("[TOOL_ERROR] Unknown system call: ${call.type}")
            }
            onStep(ToolStepEvent(ToolStepEvent.Type.FINISHED, title, summarizeToolResult(result.observation)))
            result
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            val message = e.message ?: e::class.java.simpleName
            onStep(ToolStepEvent(ToolStepEvent.Type.FAILED, title, message))
            SingleToolResult("[TOOL_ERROR] $title failed: $message")
        }
    }

    private fun toolTitle(call: AgentSystemCall): String {
        return when (normalizeType(call.type)) {
            "get_current_time" -> "Get current time"
            "web_search" -> "Web search"
            "browser_url", "browse_url", "web_browse", "open_url" -> "Read web page"
            "python_exec", "run_python", "python" -> "Run Python"
            else -> "Run tool"
        }
    }

    private fun toolDetail(call: AgentSystemCall): String {
        return when (normalizeType(call.type)) {
            "web_search" -> call.query.orEmpty()
            "browser_url", "browse_url", "web_browse", "open_url" -> call.url.orEmpty().ifBlank { call.query.orEmpty() }
            "python_exec", "run_python", "python" ->
                call.code.orEmpty().lineSequence().firstOrNull()?.trim().orEmpty().ifBlank { "python_exec" }
            else -> call.type.orEmpty()
        }.take(240)
    }

    private fun summarizeToolResult(result: String): String {
        return when {
            result.contains("[SEARCH_RESULT]") -> "Search completed, ${result.length} chars"
            result.contains("[BROWSE_RESULT]") -> "Page read, ${result.length} chars"
            result.contains("[PYTHON_RESULT]") -> "Python completed, ${result.length} chars"
            result.contains("[PYTHON_ERROR]") -> "Python failed"
            result.contains("[TIME_RESULT]") -> "Time read"
            result.contains("[TOOL_ERROR]") ||
                result.contains("[SEARCH_ERROR]") ||
                result.contains("[BROWSE_ERROR]") -> "Tool failed"
            else -> "Done, ${result.length} chars"
        }
    }

    private fun normalizeType(type: String?): String {
        return type.orEmpty().trim().lowercase(Locale.US)
    }

    private fun executeGetCurrentTime(): String {
        val zone = ZoneId.systemDefault()
        val now = LocalDateTime.now(zone)
        val formatted = now.format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"))
        val weekday = now.dayOfWeek.getDisplayName(TextStyle.FULL, Locale.getDefault())
        val epochMs = now.atZone(zone).toInstant().toEpochMilli()
        return """
            [TIME_RESULT]
            datetime: $formatted
            weekday: $weekday
            timezone: ${zone.id}
            epoch_ms: $epochMs
        """.trimIndent()
    }

    private suspend fun executePython(call: AgentSystemCall): SingleToolResult {
        val code = call.code.orEmpty().ifBlank { call.query.orEmpty() }
        if (code.isBlank()) return SingleToolResult("[PYTHON_ERROR] Empty Python code.")
        val result = AgenticPythonEngine.execute(
            code = code,
            input = call.input.orEmpty(),
            timeoutMs = normalizePythonTimeout(code, call.timeoutMs)
        )
        val files = result.files.mapNotNull { file ->
            if (file.path.isBlank()) {
                null
            } else {
                ChatFileAttachment(
                    name = file.name.ifBlank { java.io.File(file.path).name },
                    path = file.path,
                    mimeType = file.mime_type,
                    sizeBytes = file.size_bytes
                )
            }
        } + resolveDeclaredOutputFiles(call)
        val observation = buildString {
            appendLine(if (result.ok) "[PYTHON_RESULT]" else "[PYTHON_ERROR]")
            appendLine("elapsed_ms: ${result.elapsed_ms}")
            if (result.stdout.isNotBlank()) {
                appendLine("stdout:")
                appendLine(result.stdout.trimEnd())
            }
            if (result.result != null) {
                appendLine("result:")
                appendLine(result.result.toString())
            }
            if (result.stderr.isNotBlank()) {
                appendLine("stderr:")
                appendLine(result.stderr.trimEnd())
            }
            if (result.error.isNotBlank()) {
                appendLine("error:")
                appendLine(result.error.trimEnd())
            }
            if (files.isNotEmpty()) {
                appendLine("files:")
                files.forEach { file ->
                    appendLine("- ${file.name} | ${file.path} | ${file.mimeType} | ${file.sizeBytes} bytes")
                }
            }
        }.trim()
        return SingleToolResult(observation, files)
    }

    private fun resolveDeclaredOutputFiles(call: AgentSystemCall): List<ChatFileAttachment> {
        val references = buildList {
            addAll(call.outputFiles.orEmpty())
            addAll(call.generatedFiles.orEmpty())
            addAll(call.expectedOutputs.orEmpty())
            addAll(call.files.orEmpty())
        }
        return resolveWorkspaceFileReferences(references)
    }

    private fun resolveWorkspaceFileReference(rawReference: String): ChatFileAttachment? {
        val workspace = runCatching { AgenticPythonEngine.workspaceDir().canonicalFile }.getOrNull()
            ?: return null
        val decoded = runCatching { URLDecoder.decode(rawReference.trim(), "UTF-8") }
            .getOrElse { rawReference.trim() }
            .trim('"', '\'', '`', ' ', '\n', '\r', '\t', '(', ')', '[', ']', '<', '>')
        if (decoded.isBlank()) return null

        val withoutScheme = decoded
            .removePrefix("file://localhost")
            .removePrefix("file://")
            .replace('\\', '/')

        val candidates = mutableListOf<File>()
        val direct = File(withoutScheme)
        if (direct.isAbsolute) {
            candidates += direct
            candidates += File(workspace, withoutScheme.trimStart('/'))
        } else {
            candidates += File(workspace, withoutScheme)
        }
        candidates += File(workspace, File(withoutScheme).name)

        val found = candidates.asSequence()
            .mapNotNull { candidate ->
                runCatching { candidate.canonicalFile }.getOrNull()
            }
            .firstOrNull { candidate ->
                candidate.isFile && candidate.path.startsWith(workspace.path + File.separator)
            } ?: findWorkspaceFileByName(workspace, File(withoutScheme).name)
            ?: return null

        return ChatFileAttachment(
            name = found.name,
            path = found.absolutePath,
            mimeType = guessMimeType(found.name),
            sizeBytes = found.length()
        )
    }

    private fun findWorkspaceFileByName(workspace: File, name: String): File? {
        if (name.isBlank() || !workspace.exists()) return null
        return workspace.walkTopDown()
            .onEnter { dir -> dir.name != "__pycache__" && dir.name != "python" }
            .filter { it.isFile && it.name == name }
            .maxByOrNull { it.lastModified() }
            ?.canonicalFile
            ?.takeIf { it.path.startsWith(workspace.path + File.separator) }
    }

    private fun normalizePythonTimeout(code: String, requestedTimeoutMs: Long?): Long {
        val requested = requestedTimeoutMs ?: 15_000L
        val needsDataPackageWarmup = listOf(
            "import pandas",
            "from pandas",
            "import numpy",
            "from numpy",
            "import openpyxl",
            "from openpyxl",
            "read_excel(",
            "write_excel(",
            ".to_excel("
        ).any { code.contains(it) }
        val minimum = if (needsDataPackageWarmup) 15_000L else 1_000L
        return requested.coerceAtLeast(minimum).coerceAtMost(30_000L)
    }

    private fun guessMimeType(name: String): String {
        val lower = name.lowercase(Locale.US)
        return when {
            lower.endsWith(".xlsx") -> "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            lower.endsWith(".xls") -> "application/vnd.ms-excel"
            lower.endsWith(".csv") -> "text/csv"
            lower.endsWith(".pdf") -> "application/pdf"
            lower.endsWith(".txt") -> "text/plain"
            lower.endsWith(".json") -> "application/json"
            lower.endsWith(".png") -> "image/png"
            lower.endsWith(".jpg") || lower.endsWith(".jpeg") -> "image/jpeg"
            lower.endsWith(".py") -> "text/x-python"
            else -> "application/octet-stream"
        }
    }

    private suspend fun executeWebSearch(query: String): String = withContext(Dispatchers.IO) {
        if (query.isBlank()) return@withContext "[SEARCH_ERROR] Empty query."
        val result = withTimeoutOrNull(12_000L) {
            searchBing(query)
        }
        if (result.isNullOrEmpty()) {
            "[SEARCH_ERROR] No search results were returned."
        } else {
            buildString {
                appendLine("[SEARCH_RESULT]")
                appendLine("query: $query")
                appendLine("source: Bing")
                result.take(MAX_SEARCH_RESULTS).forEachIndexed { index, hit ->
                    appendLine("${index + 1}. ${hit.title}")
                    appendLine("   url: ${hit.url}")
                    if (hit.snippet.isNotBlank()) appendLine("   snippet: ${hit.snippet}")
                }
            }.trim()
        }
    }

    private fun searchBing(query: String): List<SearchHit> {
        val encoded = URLEncoder.encode(query, "UTF-8")
        val url = URL("https://www.bing.com/search?q=$encoded&form=QBRE&pq=$encoded&qs=n&sp=-1&lq=0")
        Log.d(TAG, "BING-REQ: $url")
        val conn = openHttp(url)
        val html = try {
            readResponseBody(conn)
        } finally {
            conn.disconnect()
        }
        val results = parseBingHtml(html)
        Log.d(TAG, "BING-RESULTS: ${results.size}")
        return results
    }

    private suspend fun executeBrowseUrl(rawUrl: String): String = withContext(Dispatchers.IO) {
        val normalized = normalizeHttpUrl(rawUrl)
            ?: return@withContext "[BROWSE_ERROR] Invalid URL; only http/https pages are supported."
        val text = withTimeoutOrNull(15_000L) {
            fetchUrlText(normalized)
        }
        if (text.isNullOrBlank()) {
            "[BROWSE_ERROR] Page content could not be read."
        } else {
            "[BROWSE_RESULT]\n$text"
        }
    }

    private fun fetchUrlText(url: String): String {
        val parsed = URL(url)
        val conn = openHttp(parsed)
        val html = try {
            readResponseBody(conn)
        } finally {
            conn.disconnect()
        }
        val title = extractHtmlTitle(html)
        val text = htmlToReadableText(html)
        return buildString {
            appendLine("final_url: $url")
            if (title.isNotBlank()) appendLine("title: $title")
            appendLine("text:")
            append(text.take(MAX_PAGE_CHARS))
        }
    }

    private fun openHttp(url: URL): HttpURLConnection {
        return (url.openConnection() as HttpURLConnection).apply {
            connectTimeout = 8_000
            readTimeout = 10_000
            instanceFollowRedirects = true
            setRequestProperty("User-Agent", USER_AGENT)
            setRequestProperty("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")
            setRequestProperty("Accept-Language", "zh-CN,zh;q=0.9,en;q=0.8")
            setRequestProperty("Accept-Encoding", "gzip")
            setRequestProperty("Cache-Control", "no-cache")
        }
    }

    private fun readResponseBody(conn: HttpURLConnection): String {
        val stream = runCatching {
            if (conn.responseCode >= 400) conn.errorStream else conn.inputStream
        }.getOrNull() ?: return ""
        val input = if (conn.contentEncoding.equals("gzip", ignoreCase = true)) {
            GZIPInputStream(stream)
        } else {
            stream
        }
        return input.use { readText(it, responseCharset(conn.contentType)) }
    }

    private fun readText(input: InputStream, charset: String): String {
        return BufferedReader(InputStreamReader(input, charset)).use { it.readText() }
    }

    private fun responseCharset(contentType: String?): String {
        val match = Regex("charset=([^;]+)", RegexOption.IGNORE_CASE).find(contentType.orEmpty())
        return match?.groupValues?.getOrNull(1)?.trim('"', '\'', ' ')?.ifBlank { null } ?: "UTF-8"
    }

    private fun parseBingHtml(html: String): List<SearchHit> {
        val results = mutableListOf<SearchHit>()
        val itemRegex = Regex("<li[^>]*class=\"[^\"]*b_algo[^\"]*\"[\\s\\S]*?</li>", RegexOption.IGNORE_CASE)
        for (item in itemRegex.findAll(html)) {
            if (results.size >= MAX_SEARCH_RESULTS) break
            val block = item.value
            val link = Regex("<h2[^>]*>[\\s\\S]*?<a[^>]*href=\"([^\"]+)\"[^>]*>([\\s\\S]*?)</a>", RegexOption.IGNORE_CASE)
                .find(block)
            val url = normalizeBingHref(link?.groupValues?.getOrNull(1).orEmpty())
            val title = stripAll(link?.groupValues?.getOrNull(2).orEmpty())
            val snippet = Regex("<p[^>]*>([\\s\\S]*?)</p>", RegexOption.IGNORE_CASE)
                .find(block)
                ?.groupValues
                ?.getOrNull(1)
                ?.let(::stripAll)
                .orEmpty()
            if (title.isNotBlank() && url.isNotBlank()) {
                results.add(SearchHit(title, url, snippet))
            }
        }
        return results
    }

    private fun normalizeBingHref(raw: String): String {
        val decoded = htmlDecode(raw)
        return if (decoded.contains("/ck/a?", ignoreCase = true) && decoded.contains("u=")) {
            val u = Regex("[?&]u=([^&]+)").find(decoded)?.groupValues?.getOrNull(1)
            if (u.isNullOrBlank()) decoded else decodeBingRedirectUrl(u) ?: decoded
        } else {
            decoded
        }
    }

    private fun decodeBingRedirectUrl(raw: String): String? {
        val decoded = runCatching { URLDecoder.decode(raw, "UTF-8") }.getOrNull() ?: return null
        if (decoded.startsWith("http://", ignoreCase = true) || decoded.startsWith("https://", ignoreCase = true)) {
            return decoded
        }
        if (decoded.startsWith("a1") && decoded.length > 2) {
            return runCatching {
                String(
                    Base64.decode(decoded.substring(2), Base64.URL_SAFE or Base64.NO_PADDING or Base64.NO_WRAP),
                    Charsets.UTF_8
                )
            }.getOrNull()?.takeIf {
                it.startsWith("http://", ignoreCase = true) || it.startsWith("https://", ignoreCase = true)
            }
        }
        return null
    }

    private fun normalizeHttpUrl(rawUrl: String): String? {
        val trimmed = rawUrl.trim()
        if (trimmed.isBlank()) return null
        val withScheme = if (
            trimmed.startsWith("http://", ignoreCase = true) ||
            trimmed.startsWith("https://", ignoreCase = true)
        ) {
            trimmed
        } else {
            "https://$trimmed"
        }
        val parsed = runCatching { URL(withScheme) }.getOrNull() ?: return null
        val protocol = parsed.protocol.lowercase(Locale.US)
        if (protocol != "http" && protocol != "https") return null
        if (parsed.host.isNullOrBlank()) return null
        return parsed.toString()
    }

    private fun extractHtmlTitle(html: String): String {
        return Regex("<title[^>]*>([\\s\\S]*?)</title>", RegexOption.IGNORE_CASE)
            .find(html)
            ?.groupValues
            ?.getOrNull(1)
            ?.let(::stripAll)
            .orEmpty()
    }

    private fun htmlToReadableText(html: String): String {
        val body = Regex("<body[^>]*>([\\s\\S]*?)</body>", RegexOption.IGNORE_CASE)
            .find(html)
            ?.groupValues
            ?.getOrNull(1)
            ?: html
        val cleaned = body
            .replace(Regex("<script[\\s\\S]*?</script>", RegexOption.IGNORE_CASE), " ")
            .replace(Regex("<style[\\s\\S]*?</style>", RegexOption.IGNORE_CASE), " ")
            .replace(Regex("<noscript[\\s\\S]*?</noscript>", RegexOption.IGNORE_CASE), " ")
            .replace(Regex("<br\\s*/?>", RegexOption.IGNORE_CASE), "\n")
            .replace(Regex("</(p|div|li|tr|h[1-6]|section|article|table)>", RegexOption.IGNORE_CASE), "\n")
        return stripAll(cleaned)
    }

    private fun stripAll(html: String?): String {
        if (html.isNullOrBlank()) return ""
        return html
            .replace(Regex("<[^>]+>"), " ")
            .let(::htmlDecode)
            .replace(Regex("[\\t\\x0B\\f\\r ]+"), " ")
            .replace(Regex("\\n\\s*\\n+"), "\n")
            .trim()
    }

    private fun htmlDecode(text: String): String {
        return text
            .replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", "\"")
            .replace("&#39;", "'")
            .replace("&nbsp;", " ")
    }
}
