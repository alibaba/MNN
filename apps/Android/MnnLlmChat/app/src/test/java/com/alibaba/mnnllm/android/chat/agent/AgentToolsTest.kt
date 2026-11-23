package com.alibaba.mnnllm.android.chat.agent

import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.awaitCancellation
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.withTimeout
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.time.ZonedDateTime

@OptIn(ExperimentalCoroutinesApi::class)
class AgentToolsTest {
    @Test
    fun `HTML extraction removes scripts and local links and merges duplicate results`() {
        val html = """<html><body><script><a href="https://bad.example/">Bad script result</a></script>
            <a href="https://example.com/a">A meaningful title</a>
            <a href="https://example.com/a">A meaningful snippet</a>
            <a href="/redirect?url=https%3A%2F%2Fexample.org%2Fb">Another meaningful result</a>
            <a href="http://127.0.0.1/">Local meaningful result</a>
            <a href="javascript:alert(1)">Invalid meaningful result</a>
            <a href="https://www.bing.com/help">Engine internal result</a>
            </body></html>"""
        val hits = AgentHtml.searchHits(html, "https://www.bing.com/search?q=MNN")
        assertEquals(2, hits.size)
        assertTrue(hits[0].text.contains("snippet"))
        assertEquals("https://example.org/b", hits[1].url)
        val (title, body) = AgentHtml.readablePage(AgentFetchedPage("https://example.com/",
            "<title>A &amp; B</title><body><script>evil</script><nav>menu</nav><p>Hello</p><p>${"x".repeat(4000)}</p></body>", true))
        assertEquals("A & B", title)
        assertFalse(body.contains("evil"))
        assertFalse(body.contains("menu"))
        assertEquals(3000, body.length)
    }

    @Test
    fun `search uses only current request keywords and fallback engines without network in tests`() = runTest {
        val consent = AgentConsentController()
        val revision = consent.confirmEnable().revision
        val requests = mutableListOf<String>()
        val fetcher = AgentPageFetcher { url, _ ->
            requests.add(url)
            val body = if (requests.size == 1) "<body>no hits</body>" else
                """<a href="https://example.com/a">First result title</a><a href="https://example.org/b">Second result title</a>"""
            AgentFetchedPage(url, body, true)
        }
        val result = AgentTools(consent, fetcher).execute(ValidatedAgentCall(AgentTool.SEARCH, "private history"),
            AgentToolContext("请搜索 MNN Android", revision))
        assertTrue(result.success)
        assertEquals(2, requests.size)
        assertTrue(requests.first().contains("MNN"))
        assertFalse(requests.any { it.contains("private") })
        assertEquals(2, result.searchHits.size)
        assertTrue(result.observation.contains("source: 360"))
    }

    @Test
    fun `local time and calculator need no network while disabled network is denied`() = runTest {
        val consent = AgentConsentController()
        val revision = consent.confirmEnable(networkAllowed = false).revision
        var fetched = false
        val tools = AgentTools(consent, AgentPageFetcher { _, _ -> fetched = true; error("Must not fetch") },
            now = { ZonedDateTime.parse("2026-11-22T11:30:01+08:00[Asia/Shanghai]") })
        val time = tools.execute(ValidatedAgentCall(AgentTool.TIME), AgentToolContext("time", revision))
        assertTrue(time.observation.contains("datetime: 2026-11-22 11:30:01"))
        assertTrue(time.observation.contains("weekday: Sunday"))
        assertTrue(time.observation.contains("timezone: Asia/Shanghai"))
        assertTrue(tools.execute(ValidatedAgentCall(AgentTool.CALCULATOR, "2*3"), AgentToolContext("calc", revision)).success)
        assertFalse(tools.execute(ValidatedAgentCall(AgentTool.SEARCH, "MNN"), AgentToolContext("MNN", revision)).success)
        assertFalse(fetched)
    }

    @Test
    fun `fetcher timeout is a failed tool rather than cancellation of its caller`() = runTest {
        val consent = AgentConsentController()
        val revision = consent.confirmEnable().revision
        val tools = AgentTools(consent, AgentPageFetcher { _, _ -> withTimeout(1) { awaitCancellation() } })
        val result = tools.execute(ValidatedAgentCall(AgentTool.READ_PAGE, "https://example.com/"), AgentToolContext("read", revision))
        assertFalse(result.success)
        assertTrue(result.observation.contains("timed out"))
    }
}
