package com.alibaba.mnnllm.android.chat

import com.alibaba.mnnllm.android.chat.model.ChatDataItem
import com.alibaba.mnnllm.android.chat.chatlist.ChatViewHolders
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class ChatSimpleTest {

    @Test
    fun `test ChatDataItem basic creation`() {
        // Given
        val time = "2024-01-01 10:00:00"
        val type = ChatViewHolders.USER
        val text = "Hello, world!"
        
        // When
        val chatDataItem = ChatDataItem(time, type, text)
        
        // Then
        assertEquals(time, chatDataItem.time)
        assertEquals(type, chatDataItem.type)
        assertEquals(text, chatDataItem.text)
        assertEquals(text, chatDataItem.displayText)
        assertFalse(chatDataItem.loading)
        assertTrue(chatDataItem.showThinking)
    }

    @Test
    fun `test assistant answer streaming all split points and copy`() {
        val body = "完整回答\n保留正文。"
        val input = "<answer>$body</answer>"
        for (split in 0..input.length) {
            val processor = GenerateResultProcessor()
            processor.generateBegin()
            val item = ChatDataItem(ChatViewHolders.ASSISTANT)
            item.loading = true
            var previous = 0
            for (end in listOf(split, input.length)) {
                processor.process(input.substring(previous, end))
                item.thinkingText = processor.getThinkingContent()
                item.displayText = processor.getNormalOutput()
                item.text = processor.getRawResult()
                val expected = body.take((end - "<answer>".length).coerceIn(0, body.length))
                assertEquals("split=$split end=$end", expected, item.displayText)
                assertEquals(expected, item.displayText ?: item.text ?: "")
                assertEquals(input.substring(0, end), item.text)
                previous = end
            }
            processor.process(null)
            item.displayText = processor.getNormalOutput()
            item.loading = false
            assertEquals(body, item.displayText)
            assertEquals(body, item.displayText ?: item.text ?: "")
            assertEquals(input, item.text)
            val history = ChatDataItem(item.time, item.type, item.text)
            history.displayText = item.displayText
            assertEquals(body, history.displayText)
            assertEquals(input, history.text)
        }
    }

    @Test
    fun `test assistant legacy history and final response assignment`() {
        val body = "\n正文 <answer>字面标签</answer> 保留\n`<answer>代码</answer>`\n"
        val raw = " \t<answer>$body</answer>\r\n"
        val expected = " \t$body\r\n"
        val legacyHistory = ChatDataItem("2024-01-01 10:00:00", ChatViewHolders.ASSISTANT, raw)
        assertEquals(expected, legacyHistory.displayText)
        assertEquals(raw, legacyHistory.text)
        legacyHistory.displayText = raw
        assertEquals(expected, legacyHistory.displayText)
        assertEquals(expected, legacyHistory.displayText ?: legacyHistory.text ?: "")
        val finalResponse = ChatDataItem(ChatViewHolders.ASSISTANT)
        finalResponse.text = raw
        finalResponse.displayText = raw
        assertEquals(expected, finalResponse.displayText)
        assertEquals(raw, finalResponse.text)
    }

    @Test
    fun `test assistant normalized history is not stripped twice`() {
        val body = "<answer>字面内层标签</answer>"
        for (prefix in listOf("", "<think>已有思考内容</think>")) {
            for (normal in listOf("<answer>$body</answer>", " \t<answer>$body</answer>\r\n")) {
                val raw = prefix + normal
                val expected = if (normal.startsWith(" \t")) " \t$body\r\n" else body
                val item = ChatDataItem("2024-01-01 10:00:00", ChatViewHolders.ASSISTANT, raw)
                item.displayText = normal
                assertEquals(expected, item.displayText)
                val savedDisplay = item.displayText
                val restored = ChatDataItem(item.time, item.type, item.text)
                restored.displayText = savedDisplay
                repeat(3) {
                    assertEquals(expected, restored.displayText)
                    assertEquals(expected, restored.displayText ?: restored.text ?: "")
                    assertEquals(raw, restored.text)
                }
            }
        }
    }

    @Test
    fun `test partial answer candidates are flushed as literal text when finished`() {
        for (raw in listOf("<", "<a", "<ans", "<answer", "<answer>正文</ans")) {
            val item = ChatDataItem(ChatViewHolders.ASSISTANT)
            item.loading = true
            item.text = raw
            item.displayText = raw
            assertEquals(if (raw.startsWith("<answer>")) "正文" else "", item.displayText)
            item.loading = false
            assertEquals(if (raw.startsWith("<answer>")) "正文</ans" else raw, item.displayText)
            assertEquals(raw, item.text)
        }
    }

    @Test
    fun `test user header and code answer literals are preserved`() {
        val literal = "<answer>用户字面示例</answer>"
        for (type in listOf(ChatViewHolders.USER, ChatViewHolders.HEADER)) {
            val item = ChatDataItem("2024-01-01 10:00:00", type, literal)
            for (loading in listOf(true, false)) {
                item.loading = loading
                assertEquals(literal, item.displayText)
                item.displayText = "<ans"
                assertEquals("<ans", item.displayText)
                item.displayText = literal
                assertEquals(literal, item.text)
            }
        }
        for (literalCode in listOf(
            "`<answer>字面示例</answer>`", "```xml\n<answer>字面示例</answer>\n```",
            "~~~xml\n<answer>字面示例</answer>\n~~~", "正文 <answer>字面示例</answer>"
        )) {
            val item = ChatDataItem("2024-01-01 10:00:00", ChatViewHolders.ASSISTANT, literalCode)
            for (loading in listOf(true, false)) {
                item.loading = loading
                assertEquals(literalCode, item.displayText)
                assertEquals(literalCode, item.displayText ?: item.text ?: "")
                assertEquals(literalCode, item.text)
            }
        }
    }

    @Test
    fun `test qwen thinking display remains separate and raw unchanged`() {
        val processor = GenerateResultProcessor()
        processor.generateBegin()
        val raw = "<think>Qwen3思考</think>正常回答 `标签 <answer>示例</answer>`"
        processor.process(raw)
        processor.process(null)
        val item = ChatDataItem(ChatViewHolders.ASSISTANT)
        item.thinkingText = processor.getThinkingContent()
        item.displayText = processor.getNormalOutput()
        item.text = processor.getRawResult()
        assertEquals("Qwen3思考\n", item.thinkingText)
        assertEquals("正常回答 `标签 <answer>示例</answer>`", item.displayText)
        assertEquals(raw, item.text)
        for (split in 0..raw.length) {
            val splitProcessor = GenerateResultProcessor()
            splitProcessor.generateBegin()
            splitProcessor.process(raw.substring(0, split))
            splitProcessor.process(raw.substring(split))
            splitProcessor.process(null)
            val splitItem = ChatDataItem(ChatViewHolders.ASSISTANT)
            splitItem.thinkingText = splitProcessor.getThinkingContent()
            splitItem.displayText = splitProcessor.getNormalOutput()
            splitItem.text = splitProcessor.getRawResult()
            assertEquals("split=$split", splitProcessor.getNormalOutput(), splitItem.displayText)
            assertEquals(splitProcessor.getThinkingContent(), splitItem.thinkingText)
            assertEquals(raw, splitItem.text)
        }
    }

    @Test
    fun `test ChatDataItem with type only`() {
        // Given
        val type = ChatViewHolders.ASSISTANT
        
        // When
        val chatDataItem = ChatDataItem(type)
        
        // Then
        assertEquals(type, chatDataItem.type)
        assertNull(chatDataItem.time)
        assertNull(chatDataItem.text)
        assertEquals("", chatDataItem.displayText)
    }

    @Test
    fun `test ChatDataItem properties`() {
        // Given
        val chatDataItem = ChatDataItem(ChatViewHolders.USER)
        
        // When - set displayText
        chatDataItem.displayText = "Custom display text"
        
        // Then
        assertEquals("Custom display text", chatDataItem.displayText)
        
        // When - set loading
        chatDataItem.loading = true
        
        // Then
        assertTrue(chatDataItem.loading)
    }

    @Test
    fun `test ChatDataItem toggleThinking`() {
        // Given
        val chatDataItem = ChatDataItem(ChatViewHolders.USER)
        chatDataItem.showThinking = true
        
        // When
        chatDataItem.toggleThinking()
        
        // Then
        assertFalse(chatDataItem.showThinking)
        
        // When - toggle again
        chatDataItem.toggleThinking()
        
        // Then
        assertTrue(chatDataItem.showThinking)
    }

    @Test
    fun `test ChatDataItem createImageInputData`() {
        // Given
        val timeString = "2024-01-01 10:00:00"
        val text = "Check out this image"
        val imageUri = android.net.Uri.parse("file:///test/image.jpg")
        
        // When
        val result = ChatDataItem.createImageInputData(timeString, text, imageUri)
        
        // Then
        assertEquals(timeString, result.time)
        assertEquals(ChatViewHolders.USER, result.type)
        assertEquals(text, result.text)
        assertEquals(imageUri, result.imageUri)
    }

    @Test
    fun `test ChatDataItem createAudioInputData`() {
        // Given
        val timeString = "2024-01-01 10:00:00"
        val text = "Voice message"
        val audioPath = "/test/audio.mp3"
        val duration = 5.5f
        
        // When
        val result = ChatDataItem.createAudioInputData(timeString, text, audioPath, duration)
        
        // Then
        assertEquals(timeString, result.time)
        assertEquals(ChatViewHolders.USER, result.type)
        assertEquals(text, result.text)
        assertEquals(duration, result.audioDuration, 0.01f)
    }

    @Test
    fun `test ChatViewHolders constants`() {
        // Then
        assertEquals(0, ChatViewHolders.HEADER)
        assertEquals(1, ChatViewHolders.ASSISTANT)
        assertEquals(2, ChatViewHolders.USER)
    }

    @Test
    fun `test ChatTestUtils createUserMessage`() {
        // When
        val userMessage = ChatTestUtils.createUserMessage("Hello")
        
        // Then
        assertEquals(ChatViewHolders.USER, userMessage.type)
        assertEquals("Hello", userMessage.text)
        assertEquals("2024-01-01 10:00:00", userMessage.time)
    }

    @Test
    fun `test ChatTestUtils createAssistantMessage`() {
        // When
        val assistantMessage = ChatTestUtils.createAssistantMessage("Hi there!")
        
        // Then
        assertEquals(ChatViewHolders.ASSISTANT, assistantMessage.type)
        assertEquals("Hi there!", assistantMessage.text)
        assertEquals("2024-01-01 10:01:00", assistantMessage.time)
    }

    @Test
    fun `test ChatTestUtils createTestChatHistory`() {
        // When
        val chatHistory = ChatTestUtils.createTestChatHistory()
        
        // Then
        assertEquals(4, chatHistory.size)
        assertEquals(ChatViewHolders.USER, chatHistory[0].type)
        assertEquals(ChatViewHolders.ASSISTANT, chatHistory[1].type)
        assertEquals(ChatViewHolders.USER, chatHistory[2].type)
        assertEquals(ChatViewHolders.ASSISTANT, chatHistory[3].type)
    }

    @Test
    fun `test ChatTestUtils createTestGenerateResult`() {
        // When
        val result = ChatTestUtils.createTestGenerateResult("Test response", "Generated in 2.5 seconds")
        
        // Then
        assertEquals("Test response", result["response"])
        assertEquals("Generated in 2.5 seconds", result["benchmark"])
        assertEquals(150, result["tokens"])
        assertEquals(2.5, result["time"])
    }

    @Test
    fun `test ChatTestUtils assertChatDataItemValid`() {
        // Given
        val chatDataItem = ChatDataItem("2024-01-01 10:00:00", ChatViewHolders.USER, "Hello")
        
        // When & Then - should not throw exception
        ChatTestUtils.assertChatDataItemValid(
            chatDataItem,
            ChatViewHolders.USER,
            "Hello",
            "2024-01-01 10:00:00"
        )
    }

    @Test
    fun `test ChatTestUtils assertGenerateResultValid`() {
        // Given
        val result = ChatTestUtils.createTestGenerateResult("Test response", "Generated in 2.5 seconds")

        // When & Then - should not throw exception
        ChatTestUtils.assertGenerateResultValid(result, "Test response")
    }

    @Test
    fun `test ChatTestUtils createTestSessionId`() {
        // When
        val sessionId1 = ChatTestUtils.createTestSessionId()
        val sessionId2 = ChatTestUtils.createTestSessionId()
        
        // Then
        assertTrue(sessionId1.startsWith("test-session-"))
        assertTrue(sessionId2.startsWith("test-session-"))
        assertNotEquals(sessionId1, sessionId2)
    }

    @Test
    fun `test ChatTestUtils createTestModelId`() {
        // When
        val modelId1 = ChatTestUtils.createTestModelId("huggingface")
        val modelId2 = ChatTestUtils.createTestModelId("openai")
        
        // Then
        assertTrue(modelId1.startsWith("huggingface/test-model-"))
        assertTrue(modelId2.startsWith("openai/test-model-"))
        assertNotEquals(modelId1, modelId2)
    }

    @Test
    fun `test ChatDataItem with different types`() {
        // Test USER type
        val userItem = ChatDataItem(ChatViewHolders.USER)
        assertEquals(ChatViewHolders.USER, userItem.type)
        
        // Test ASSISTANT type
        val assistantItem = ChatDataItem(ChatViewHolders.ASSISTANT)
        assertEquals(ChatViewHolders.ASSISTANT, assistantItem.type)
        
        // Test HEADER type
        val headerItem = ChatDataItem(ChatViewHolders.HEADER)
        assertEquals(ChatViewHolders.HEADER, headerItem.type)
    }

    @Test
    fun `test ChatDataItem with null values`() {
        // Given
        val chatDataItem = ChatDataItem("2024-01-01 10:00:00", ChatViewHolders.USER, null)
        
        // Then
        assertEquals("2024-01-01 10:00:00", chatDataItem.time)
        assertEquals(ChatViewHolders.USER, chatDataItem.type)
        assertNull(chatDataItem.text)
        assertEquals("", chatDataItem.displayText)
    }

    @Test
    fun `test ChatDataItem displayText with null`() {
        // Given
        val chatDataItem = ChatDataItem(ChatViewHolders.USER)
        
        // When
        chatDataItem.displayText = null
        
        // Then
        assertEquals("", chatDataItem.displayText)
    }

    @Test
    fun `test ChatDataItem audioDuration property`() {
        // Given
        val chatDataItem = ChatDataItem(ChatViewHolders.USER)
        val duration = 10.5f
        
        // When
        chatDataItem.audioDuration = duration
        
        // Then
        assertEquals(duration, chatDataItem.audioDuration, 0.01f)
    }

    @Test
    fun `test ChatDataItem benchmarkInfo property`() {
        // Given
        val chatDataItem = ChatDataItem(ChatViewHolders.USER)
        val benchmarkInfo = "Generated in 2.5 seconds"
        
        // When
        chatDataItem.benchmarkInfo = benchmarkInfo
        
        // Then
        assertEquals(benchmarkInfo, chatDataItem.benchmarkInfo)
    }

    @Test
    fun `test ChatDataItem thinkingFinishedTime property`() {
        // Given
        val chatDataItem = ChatDataItem(ChatViewHolders.USER)
        val thinkingTime = 1640995200000L
        
        // When
        chatDataItem.thinkingFinishedTime = thinkingTime
        
        // Then
        assertEquals(thinkingTime, chatDataItem.thinkingFinishedTime)
    }

    @Test
    fun `test ChatDataItem thinkingText property`() {
        // Given
        val chatDataItem = ChatDataItem(ChatViewHolders.USER)
        val thinkingText = "I'm thinking about this..."
        
        // When
        chatDataItem.thinkingText = thinkingText
        
        // Then
        assertEquals(thinkingText, chatDataItem.thinkingText)
    }

    @Test
    fun `test ChatDataItem hasOmniAudio property`() {
        // Given
        val chatDataItem = ChatDataItem(ChatViewHolders.USER)
        
        // When
        chatDataItem.hasOmniAudio = true
        
        // Then
        assertTrue(chatDataItem.hasOmniAudio)
        
        // When
        chatDataItem.hasOmniAudio = false
        
        // Then
        assertFalse(chatDataItem.hasOmniAudio)
    }
} 