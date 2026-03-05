package com.alibaba.mnnllm.api.openai.network.compat

import TextContent
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ProtocolCompatibilityTest {

    @Test
    fun buildBaseUrl_usesHttpsWhenEnabled() {
        val httpsUrl = EndpointUrlBuilder.buildBaseUrl("127.0.0.1", 8080, useHttps = true)
        val httpUrl = EndpointUrlBuilder.buildBaseUrl("127.0.0.1", 8080, useHttps = false)

        assertEquals("https://127.0.0.1:8080", httpsUrl)
        assertEquals("http://127.0.0.1:8080", httpUrl)
    }

    @Test
    fun toOpenAIRequest_mapsAnthropicTextMessages() {
        val request = AnthropicMessagesRequest(
            model = "mnn-local",
            messages = listOf(
                AnthropicMessage(
                    role = "user",
                    content = listOf(AnthropicContentBlock(type = "text", text = "hello from anthropic"))
                )
            ),
            stream = false
        )

        val openAiRequest = AnthropicAdapter.toOpenAIRequest(request)

        assertEquals("mnn-local", openAiRequest.model)
        assertEquals(1, openAiRequest.messages.size)
        val firstMessage = openAiRequest.messages.first()
        assertEquals("user", firstMessage.role)
        assertTrue(firstMessage.content is TextContent)
        assertEquals("hello from anthropic", (firstMessage.content as TextContent).text)
    }
}
