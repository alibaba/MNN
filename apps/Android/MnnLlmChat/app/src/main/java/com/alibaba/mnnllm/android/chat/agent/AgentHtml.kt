package com.alibaba.mnnllm.android.chat.agent

import org.jsoup.Jsoup
import org.jsoup.nodes.Element

data class AgentSearchHit(val text: String, val url: String)

object AgentHtml {
    const val MAX_SEARCH_RESULTS = 5
    const val MAX_PAGE_CHARACTERS = 3000
    private val policy = AgentUrlPolicy()

    fun searchHits(html: String, baseUrl: String): List<AgentSearchHit> {
        require(html.length <= AgentHttpClient.MAX_RESPONSE_BYTES)
        val base = policy.validate(baseUrl)
        val engineDomain = base.topPrivateDomain() ?: base.host
        val document = Jsoup.parse(html, base.toString())
        document.select("script,style,noscript,svg,iframe,form").remove()
        val hits = linkedMapOf<String, MutableList<String>>()
        for (anchor in document.select("a[href]").take(2000)) {
            val text = anchor.text().trim().take(240)
            if (text.length < 8) continue
            val destination = destination(anchor, base.toString()) ?: continue
            val url = try { policy.validate(destination) } catch (_: IllegalArgumentException) { continue }
            if (url.host == engineDomain || url.host.endsWith(".$engineDomain")) continue
            val parts = hits.getOrPut(url.toString()) { mutableListOf() }
            if (text !in parts && parts.size < 3) parts.add(text)
        }
        return hits.entries.take(MAX_SEARCH_RESULTS).map { AgentSearchHit(it.value.joinToString(" — ").take(240), it.key) }
    }

    private fun destination(anchor: Element, base: String): String? {
        for (attribute in listOf("mu", "data-url", "data-mdurl")) {
            val value = anchor.attr(attribute)
            if (value.startsWith("http://", true) || value.startsWith("https://", true)) return value
        }
        val href = anchor.attr("href")
        val resolved = policy.validate(base).resolve(href) ?: return null
        for (key in listOf("url", "u", "target")) {
            val value = resolved.queryParameter(key)
            if (value != null && (value.startsWith("http://", true) || value.startsWith("https://", true))) return value
        }
        return resolved.toString()
    }

    fun readablePage(page: AgentFetchedPage): Pair<String, String> {
        if (!page.isHtml) return "" to page.text.trim().take(MAX_PAGE_CHARACTERS)
        require(page.text.length <= AgentHttpClient.MAX_RESPONSE_BYTES)
        val document = Jsoup.parse(page.text, page.url)
        val title = document.title().trim().take(160)
        document.select("script,style,noscript,svg,nav,footer,header,iframe,form,object,embed,template").remove()
        val body = document.body()
        body.select("br").append("\n")
        body.select("p,div,li,tr,h1,h2,h3,h4,h5,h6,section,article,table").append("\n")
        val text = body.wholeText().replace(Regex("[ \\t\\r\\u00a0]+"), " ")
            .replace(Regex(" *\\n *"), "\n").replace(Regex("\\n{2,}"), "\n").trim()
        return title to text.take(MAX_PAGE_CHARACTERS)
    }
}
