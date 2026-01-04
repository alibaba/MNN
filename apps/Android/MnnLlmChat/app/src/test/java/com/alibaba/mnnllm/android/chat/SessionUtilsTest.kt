package com.alibaba.mnnllm.android.chat

import android.net.Uri
import com.alibaba.mnnllm.android.chat.chatlist.ChatViewHolders
import com.alibaba.mnnllm.android.chat.model.ChatDataItem
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class SessionUtilsTest {

    @Test
    fun `generateSessionName with plain text should return text as is`() {
        // Given
        val chatDataItem = ChatDataItem(ChatViewHolders.USER).apply {
            text = "Hello, this is a test message"
        }

        // When
        val result = SessionUtils.generateSessionName(chatDataItem)

        // Then
        assertEquals("Hello, this is a test message", result)
    }

    @Test
    fun `generateSessionName with audio should add Audio prefix`() {
        // Given
        val chatDataItem = ChatDataItem(ChatViewHolders.USER).apply {
            text = "Voice message"
            audioUri = Uri.parse("file:///test/audio.wav")
        }

        // When
        val result = SessionUtils.generateSessionName(chatDataItem)

        // Then
        assertEquals("[Audio] Voice message", result)
    }

    @Test
    fun `generateSessionName with image should add Image prefix`() {
        // Given
        val chatDataItem = ChatDataItem(ChatViewHolders.USER).apply {
            text = "Image description"
            imageUri = Uri.parse("file:///test/image.jpg")
        }

        // When
        val result = SessionUtils.generateSessionName(chatDataItem)

        // Then
        assertEquals("[Image] Image description", result)
    }

    @Test
    fun `generateSessionName with video should add Video prefix`() {
        // Given
        val chatDataItem = ChatDataItem(ChatViewHolders.USER).apply {
            text = "Video content"
            videoUri = Uri.parse("file:///test/video.mp4")
        }

        // When
        val result = SessionUtils.generateSessionName(chatDataItem)

        // Then
        assertEquals("[Video] Video content", result)
    }

    @Test
    fun `generateSessionName with all media types should add all prefixes`() {
        // Given
        val chatDataItem = ChatDataItem(ChatViewHolders.USER).apply {
            text = "Multimedia message"
            audioUri = Uri.parse("file:///test/audio.wav")
            imageUri = Uri.parse("file:///test/image.jpg")
            videoUri = Uri.parse("file:///test/video.mp4")
        }

        // When
        val result = SessionUtils.generateSessionName(chatDataItem)

        // Then
        assertEquals("[Video] [Image] [Audio] Multimedia message", result)
    }

    @Test
    fun `generateSessionName with text longer than 100 chars should truncate`() {
        // Given
        val longText = "a".repeat(150)
        val chatDataItem = ChatDataItem(ChatViewHolders.USER).apply {
            text = longText
        }

        // When
        val result = SessionUtils.generateSessionName(chatDataItem)

        // Then
        assertEquals(100, result.length)
        assertEquals("a".repeat(100), result)
    }

    @Test
    fun `generateSessionName with exactly 100 chars should not truncate`() {
        // Given
        val text = "a".repeat(100)
        val chatDataItem = ChatDataItem(ChatViewHolders.USER).apply {
            this.text = text
        }

        // When
        val result = SessionUtils.generateSessionName(chatDataItem)

        // Then
        assertEquals(100, result.length)
        assertEquals(text, result)
    }

    @Test
    fun `generateSessionName with 99 chars should not truncate`() {
        // Given
        val text = "a".repeat(99)
        val chatDataItem = ChatDataItem(ChatViewHolders.USER).apply {
            this.text = text
        }

        // When
        val result = SessionUtils.generateSessionName(chatDataItem)

        // Then
        assertEquals(99, result.length)
        assertEquals(text, result)
    }

    @Test
    fun `generateSessionName with 101 chars should truncate to 100`() {
        // Given
        val text = "a".repeat(101)
        val chatDataItem = ChatDataItem(ChatViewHolders.USER).apply {
            this.text = text
        }

        // When
        val result = SessionUtils.generateSessionName(chatDataItem)

        // Then
        assertEquals(100, result.length)
        assertEquals("a".repeat(100), result)
    }

    @Test
    fun `generateSessionName with empty text should return empty string`() {
        // Given
        val chatDataItem = ChatDataItem(ChatViewHolders.USER).apply {
            text = ""
        }

        // When
        val result = SessionUtils.generateSessionName(chatDataItem)

        // Then
        assertEquals("", result)
    }

    @Test
    fun `generateSessionName with audio and long text should truncate including prefix`() {
        // Given
        val longText = "a".repeat(100)
        val chatDataItem = ChatDataItem(ChatViewHolders.USER).apply {
            text = longText
            audioUri = Uri.parse("file:///test/audio.wav")
        }

        // When
        val result = SessionUtils.generateSessionName(chatDataItem)

        // Then
        // "[Audio] " is 8 chars, so total would be 108, should truncate to 100
        assertEquals(100, result.length)
        assertTrue(result.startsWith("[Audio] "))
    }

    @Test
    fun `generateSessionName with special characters should preserve them`() {
        // Given
        val chatDataItem = ChatDataItem(ChatViewHolders.USER).apply {
            text = "Hello! @#$%^&*() 你好 🎉"
        }

        // When
        val result = SessionUtils.generateSessionName(chatDataItem)

        // Then
        assertEquals("Hello! @#$%^&*() 你好 🎉", result)
    }
}
