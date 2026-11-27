package com.alibaba.mnnllm.android.chat.agent

import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withTimeout
import okhttp3.Call
import okhttp3.Callback
import okhttp3.HttpUrl
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.net.Proxy
import java.nio.charset.StandardCharsets
import java.util.concurrent.TimeUnit
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

data class AgentFetchedPage(val url: String, val text: String, val isHtml: Boolean)

fun interface AgentPageFetcher {
    suspend fun fetch(url: String, consentRevision: Long): AgentFetchedPage
}

class AgentHttpClient(
    private val consent: AgentConsentController,
    private val policy: AgentUrlPolicy = AgentUrlPolicy(),
) : AgentPageFetcher {
    private val client = OkHttpClient.Builder()
        .dns(AgentPublicDns(policy))
        .proxy(Proxy.NO_PROXY)
        .followRedirects(false)
        .followSslRedirects(false)
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(10, TimeUnit.SECONDS)
        .callTimeout(15, TimeUnit.SECONDS)
        .retryOnConnectionFailure(false)
        .build()

    override suspend fun fetch(url: String, consentRevision: Long): AgentFetchedPage =
        fetchWithTransport(url, consentRevision) { fetchOnce(it, consentRevision) }

    internal suspend fun fetchWithTransport(url: String, consentRevision: Long,
        transport: suspend (HttpUrl) -> HttpResult): AgentFetchedPage = withTimeout(15000) {
        var current = policy.validate(url)
        repeat(5) { index ->
            require(consent.canUseNetwork(consentRevision)) { "Network consent is required" }
            when (val fetched = transport(current)) {
                is HttpResult.Page -> return@withTimeout fetched.page
                is HttpResult.Redirect -> {
                    require(index < 4) { "Too many redirects" }
                    val next = policy.validate(fetched.url)
                    require(!current.isHttps || next.isHttps) { "HTTPS downgrade is forbidden" }
                    current = next
                }
            }
        }
        error("Too many redirects")
    }

    internal sealed class HttpResult {
        data class Page(val page: AgentFetchedPage) : HttpResult()
        data class Redirect(val url: String) : HttpResult()
    }

    private suspend fun fetchOnce(url: HttpUrl, consentRevision: Long): HttpResult = suspendCancellableCoroutine { continuation ->
        if (!consent.canUseNetwork(consentRevision)) {
            continuation.resumeWithException(IllegalArgumentException("Network consent is required"))
            return@suspendCancellableCoroutine
        }
        val request = Request.Builder().url(url)
            .header("User-Agent", "Mozilla/5.0 (Linux; Android 8.0) AppleWebKit/537.36 Chrome/120.0 Mobile Safari/537.36 MNNChat")
            .header("Accept", "text/html,application/xhtml+xml,text/plain;q=0.9")
            .header("Accept-Language", "zh-CN,zh;q=0.9,en;q=0.8")
            .get().build()
        val call = client.newCall(request)
        continuation.invokeOnCancellation { call.cancel() }
        call.enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                if (continuation.isActive) continuation.resumeWithException(IOException("Page request failed"))
            }

            override fun onResponse(call: Call, response: Response) {
                try {
                    val result = readResponse(response, url, consentRevision) { continuation.isActive }
                    if (continuation.isActive) continuation.resume(result)
                } catch (e: Exception) {
                    if (continuation.isActive) continuation.resumeWithException(e)
                }
            }
        })
    }

    internal fun readResponse(response: Response, url: HttpUrl, consentRevision: Long,
        isActive: () -> Boolean): HttpResult = response.use {
        require(consent.canUseNetwork(consentRevision)) { "Network consent was revoked" }
        if (it.code in setOf(301, 302, 303, 307, 308)) {
            val location = it.header("Location") ?: throw IOException("Redirect has no location")
            val resolved = url.resolve(location) ?: throw IOException("Invalid redirect")
            return@use HttpResult.Redirect(resolved.toString())
        }
        if (!it.isSuccessful) throw IOException("HTTP ${it.code}")
        val body = it.body ?: throw IOException("Empty page")
        val type = body.contentType()
        val major = type?.type
        val subtype = type?.subtype
        val html = major == "text" && subtype == "html" || major == "application" && subtype == "xhtml+xml"
        require(html || (major == "text" && subtype == "plain")) { "Unsupported page content type" }
        require(body.contentLength() <= MAX_RESPONSE_BYTES) { "Page response too large" }
        val bytes = ByteArrayOutputStream()
        body.byteStream().use { stream ->
            val buffer = ByteArray(8192)
            var total = 0
            while (true) {
                if (!isActive() || !consent.canUseNetwork(consentRevision)) throw IOException("Request cancelled")
                val count = stream.read(buffer)
                if (count < 0) break
                total += count
                require(total <= MAX_RESPONSE_BYTES) { "Page response too large" }
                bytes.write(buffer, 0, count)
            }
        }
        HttpResult.Page(AgentFetchedPage(url.toString(),
            bytes.toString((type?.charset(StandardCharsets.UTF_8) ?: StandardCharsets.UTF_8).name()), html))
    }

    companion object {
        const val MAX_RESPONSE_BYTES = 512 * 1024
    }
}
