package com.alibaba.mnnllm.android.history

import android.net.Uri
import com.alibaba.mnnllm.android.chat.chatlist.ChatViewHolders
import com.alibaba.mnnllm.android.chat.model.ChatDataItem
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [28])
class HistoryDisplayDataTest {
    @Test
    fun missingAttachmentsAreOnlyRemovedFromTheDisplayCopy() {
        val missing = Uri.parse("file:///missing/image.png")
        val present = Uri.parse("file:///present/image.png")
        val original = ChatDataItem("123", ChatViewHolders.USER, "question").apply {
            imageUris = listOf(missing, present)
            audioUri = missing
            videoUri = missing
            loading = true
        }
        val display = HistoryDisplayData.prepare(listOf(original), "[missing]") { it == present }.single()
        assertNotSame(original, display)
        assertEquals(listOf(present), display.imageUris)
        assertNull(display.audioUri)
        assertNull(display.videoUri)
        assertFalse(display.loading)
        assertEquals("question\n\n[missing]", display.text)
        assertEquals(listOf(missing, present), original.imageUris)
        assertEquals(missing, original.audioUri)
        assertEquals("question", original.text)
        assertTrue(original.loading)
    }

    @Test
    fun assistantMarkdownThinkingAndMetadataArePreserved() {
        val original = ChatDataItem("123", ChatViewHolders.ASSISTANT, "raw").apply {
            displayText = "```kotlin\nval x = 1\n```"
            thinkingText = "reasoning"
            thinkingFinishedTime = 100
            benchmarkInfo = "stats"
            showThinking = false
        }
        val display = HistoryDisplayData.prepare(listOf(original), "[missing]") { true }.single()
        assertEquals(original.displayText, display.displayText)
        assertEquals(original.thinkingText, display.thinkingText)
        assertEquals(original.thinkingFinishedTime, display.thinkingFinishedTime)
        assertEquals(original.benchmarkInfo, display.benchmarkInfo)
        assertEquals(original.time, display.time)
        assertFalse(display.showThinking)
        assertFalse(display.loading)
        assertNull(display.audioPlayComponent)
    }

    @Test
    fun attachmentOnlyMessagesStillExplainUnavailableMedia() {
        val original = ChatDataItem(ChatViewHolders.ASSISTANT).apply {
            imageUri = Uri.parse("file:///missing/image.png")
        }
        val display = HistoryDisplayData.prepare(listOf(original), "[missing]") { false }.single()
        assertEquals("[missing]", display.text)
        assertEquals("[missing]", display.displayText)
        assertNotNull(original.imageUri)
    }
}
