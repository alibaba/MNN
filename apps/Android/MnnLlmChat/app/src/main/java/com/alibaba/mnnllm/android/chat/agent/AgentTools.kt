package com.alibaba.mnnllm.android.chat.agent

import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withTimeout
import okhttp3.HttpUrl.Companion.toHttpUrl
import java.time.ZonedDateTime
import java.time.format.DateTimeFormatter
import java.util.Locale

data class AgentToolContext(val userRequest: String, val consentRevision: Long)
data class AgentToolResult(
    val observation: String,
    val success: Boolean = true,
    val searchHits: List<AgentSearchHit> = emptyList(),
) {
    companion object {
        fun error(reason: String) = AgentToolResult("[TOOL_ERROR] ${reason.take(240)}", success = false)
    }
}

fun interface AgentToolExecutor {
    suspend fun execute(call: ValidatedAgentCall, context: AgentToolContext): AgentToolResult
}

class AgentTools(
    private val consent: AgentConsentController,
    private val fetcher: AgentPageFetcher = AgentHttpClient(consent),
    private val now: () -> ZonedDateTime = { ZonedDateTime.now() },
) : AgentToolExecutor {
    override suspend fun execute(call: ValidatedAgentCall, context: AgentToolContext): AgentToolResult {
        currentCoroutineContext().ensureActive()
        if (!consent.isCurrent(context.consentRevision)) return AgentToolResult.error("Agent consent is required")
        if (call.tool in setOf(AgentTool.SEARCH, AgentTool.READ_PAGE) && !consent.canUseNetwork(context.consentRevision)) {
            return AgentToolResult.error("Network consent is required")
        }
        return try {
            when (call.tool) {
                AgentTool.TIME -> AgentToolResult(currentTime(now()))
                AgentTool.CALCULATOR -> {
                    val result = AgentCalculator.observation(call.value)
                    AgentToolResult(result, success = result.startsWith("[CALC_RESULT]"))
                }
                AgentTool.SEARCH -> search(call.value, context)
                AgentTool.READ_PAGE -> {
                    val page = fetcher.fetch(call.value, context.consentRevision)
                    val (title, text) = AgentHtml.readablePage(page)
                    if (text.isEmpty()) AgentToolResult.error("Page content could not be read")
                    else AgentToolResult("[BROWSE_RESULT]\nurl: ${page.url}\ntitle: $title\ntext:\n$text")
                }
                AgentTool.MEMORY -> AgentToolResult.error("Memory must be handled by the consent-gated store")
            }
        } catch (e: TimeoutCancellationException) {
            currentCoroutineContext().ensureActive()
            AgentToolResult.error("${call.tool.wireName} timed out")
        } catch (e: CancellationException) { throw e }
        catch (e: Exception) { AgentToolResult.error("${call.tool.wireName} failed (${e.javaClass.simpleName})") }
    }

    private suspend fun search(modelQuery: String, context: AgentToolContext): AgentToolResult = withTimeout(35000) {
        val query = AgentInputPolicy.privateSearchQuery(modelQuery, context.userRequest)
        if (query.isBlank()) return@withTimeout AgentToolResult.error("No permitted search keywords in the current request")
        for ((name, base, parameter) in SEARCH_ENGINES) {
            currentCoroutineContext().ensureActive()
            if (!consent.canUseNetwork(context.consentRevision)) return@withTimeout AgentToolResult.error("Network consent was revoked")
            try {
                val url = base.toHttpUrl().newBuilder().addQueryParameter(parameter, query).build()
                val page = fetcher.fetch(url.toString(), context.consentRevision)
                val hits = AgentHtml.searchHits(page.text, url.toString())
                if (hits.size < 2) continue
                val observation = buildString {
                    append("[SEARCH_RESULT]\nquery: $query\nsource: $name")
                    hits.forEachIndexed { index, hit -> append("\n${index + 1}. ${hit.text}\n   url: ${hit.url}") }
                }
                return@withTimeout AgentToolResult(observation, searchHits = hits)
            } catch (e: TimeoutCancellationException) { currentCoroutineContext().ensureActive() }
            catch (e: CancellationException) { throw e }
            catch (_: Exception) { }
        }
        AgentToolResult.error("No search results were returned")
    }

    companion object {
        private val SEARCH_ENGINES = listOf(
            Triple("Sogou", "https://m.sogou.com/web/searchList.jsp", "keyword"),
            Triple("360", "https://m.so.com/s", "q"),
            Triple("Bing", "https://www.bing.com/search", "q"),
        )

        fun currentTime(now: ZonedDateTime = ZonedDateTime.now()): String =
            "[TIME_RESULT]\ndatetime: ${now.format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss", Locale.US))}" +
                "\nweekday: ${now.format(DateTimeFormatter.ofPattern("EEEE", Locale.US))}\ntimezone: ${now.zone.id}"
    }
}
