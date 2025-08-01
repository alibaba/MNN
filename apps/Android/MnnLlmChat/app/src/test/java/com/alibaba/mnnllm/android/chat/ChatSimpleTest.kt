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