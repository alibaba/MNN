package com.alibaba.mnnllm.android.chat.agent

import kotlinx.coroutines.runBlocking
import okhttp3.HttpUrl.Companion.toHttpUrl
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.Protocol
import okhttp3.Request
import okhttp3.Response
import okhttp3.ResponseBody
import okhttp3.ResponseBody.Companion.toResponseBody
import okio.Buffer
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.IOException

class AgentHttpClientTest {
    private val consent = AgentConsentController()
    private val revision = runBlocking { consent.confirmEnable().revision }
    private val client = AgentHttpClient(consent)
    private val url = "https://example.com/".toHttpUrl()

    private fun response(body: ResponseBody, code: Int = 200): Response = Response.Builder()
        .request(Request.Builder().url(url).build()).protocol(Protocol.HTTP_1_1).code(code).message("test").body(body).build()

    @Test
    fun `response reader enforces declared and actual body limits with no socket traffic`() {
        val bytes = ByteArray(AgentHttpClient.MAX_RESPONSE_BYTES + 1) { 'x'.code.toByte() }
        assertThrows(IllegalArgumentException::class.java) {
            client.readResponse(response(bytes.toResponseBody("text/html".toMediaType())), url, revision) { true }
        }
        val buffer = Buffer().write(bytes)
        val unknownLength = object : ResponseBody() {
            override fun contentType() = "text/html".toMediaType()
            override fun contentLength() = -1L
            override fun source() = buffer
        }
        assertThrows(IllegalArgumentException::class.java) {
            client.readResponse(response(unknownLength), url, revision) { true }
        }
    }

    @Test
    fun `text content types charset HTTP failures and cancellation are checked`() {
        val result = client.readResponse(response("<body>测试</body>".toResponseBody("text/html; charset=utf-8".toMediaType())), url, revision) { true }
        assertTrue(result is AgentHttpClient.HttpResult.Page)
        assertEquals("<body>测试</body>", (result as AgentHttpClient.HttpResult.Page).page.text)
        assertThrows(IllegalArgumentException::class.java) {
            client.readResponse(response(ByteArray(8).toResponseBody("application/octet-stream".toMediaType())), url, revision) { true }
        }
        assertThrows(IOException::class.java) {
            client.readResponse(response("text".toResponseBody("text/plain".toMediaType()), 403), url, revision) { true }
        }
        assertThrows(IOException::class.java) {
            client.readResponse(response("text".toResponseBody("text/plain".toMediaType())), url, revision) { false }
        }
    }
}
