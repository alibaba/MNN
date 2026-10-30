package com.alibaba.mnnllm.api.openai.network.compat

import TextContent
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.assertFalse
import org.junit.Assert.fail
import org.junit.Test
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

class ProtocolCompatibilityTest {
    private val json = Json { ignoreUnknownKeys = true }

    @Test
    fun ollamaTextRequestsAndResponses() {
        val chat = OllamaAdapter.parse(json.parseToJsonElement("""{"model":"test","messages":[{"role":"system","content":"rules"},{"role":"user","content":"hello"}]}""").jsonObject, true)
        assertTrue(chat.stream)
        assertEquals("rules", (chat.messages[0].content as TextContent).text)
        val generate = OllamaAdapter.parse(json.parseToJsonElement("""{"model":"test","system":"rules","prompt":"hello","stream":false}""").jsonObject, false)
        assertEquals(false, generate.stream)
        assertEquals("hello", (generate.messages[1].content as TextContent).text)
        val response = json.parseToJsonElement(OllamaAdapter.response("test", "a\n\"b", true, true, 3, 2)).jsonObject
        assertEquals("a\n\"b", response.getValue("message").jsonObject.getValue("content").jsonPrimitive.content)
        assertEquals("true", response.getValue("done").jsonPrimitive.content)
        assertEquals("2", response.getValue("eval_count").jsonPrimitive.content)
    }

    @Test
    fun ollamaRejectsUnsupportedAndInvalidInput() {
        val invalid = listOf(
            """{"model":"test","prompt":"hello","options":{"temperature":0}}""",
            """{"model":"test","prompt":"hello","raw":true}""",
            """{"model":"test","prompt":"hello","stream":"false"}""",
            """{"model":"test","prompt":null}"""
        )
        invalid.forEach { raw ->
            try {
                OllamaAdapter.parse(json.parseToJsonElement(raw).jsonObject, false)
                throw AssertionError("Expected request rejection: $raw")
            } catch (_: IllegalArgumentException) {
            }
        }
    }

    @Test
    fun ollamaAcceptsOfficialClientEmptyOptionsButRejectsTools() {
        val request = OllamaAdapter.parse(json.parseToJsonElement(
            """{"model":"test","messages":[{"role":"user","content":"hello","images":[]}],"tools":[],"options":{},"stream":false}"""
        ).jsonObject, true)
        assertEquals("hello", (request.messages.single().content as TextContent).text)
        try {
            OllamaAdapter.parse(json.parseToJsonElement(
                """{"model":"test","messages":[{"role":"user","content":"hello"}],"tools":[{"type":"function"}]}"""
            ).jsonObject, true)
            fail("Non-empty tools must be rejected")
        } catch (_: IllegalArgumentException) {
        }
    }

    @Test
    fun ollamaDoesNotReportNativeErrorsAsSuccess() {
        OllamaAdapter.checkGenerationResult(mapOf("decode_len" to 3L))
        try {
            OllamaAdapter.checkGenerationResult(mapOf("error" to "Model generation failed"))
            fail("Native errors must be reported")
        } catch (e: IllegalStateException) {
            assertEquals("Model generation failed", e.message)
        }
        val result = json.parseToJsonElement(OllamaAdapter.response("test", "text", true, false)).jsonObject
        assertFalse(result.containsKey("done_reason"))
    }

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

    @Test
    fun toOpenAIRequest_acceptsAnthropicStringContent() {
        val rawRequest = """
            {
              "model": "mnn-local",
              "messages": [
                {
                  "role": "user",
                  "content": "hello from claude client"
                }
              ],
              "stream": false
            }
        """.trimIndent()

        val request = json.decodeFromString<AnthropicMessagesRequest>(rawRequest)
        val openAiRequest = AnthropicAdapter.toOpenAIRequest(request)

        assertEquals(1, openAiRequest.messages.size)
        val firstMessage = openAiRequest.messages.first()
        assertEquals("user", firstMessage.role)
        assertTrue(firstMessage.content is TextContent)
        assertEquals("hello from claude client", (firstMessage.content as TextContent).text)
    }

    @Test
    fun toOpenAIRequest_acceptsAnthropicSystemArray() {
        val rawRequest = """
            {
              "model": "mnn-local",
              "system": [
                {"type":"text","text":"You are concise."}
              ],
              "messages": [
                {
                  "role": "user",
                  "content": "hello"
                }
              ],
              "stream": false
            }
        """.trimIndent()

        val request = json.decodeFromString<AnthropicMessagesRequest>(rawRequest)
        val openAiRequest = AnthropicAdapter.toOpenAIRequest(request)

        assertEquals(2, openAiRequest.messages.size)
        val systemMessage = openAiRequest.messages.first()
        assertEquals("system", systemMessage.role)
        assertTrue(systemMessage.content is TextContent)
        assertEquals("You are concise.", (systemMessage.content as TextContent).text)
    }
}
