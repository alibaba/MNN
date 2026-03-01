package com.alibaba.mnnllm.android.chat.input

import com.alibaba.mnnllm.android.chat.DEFAULT_SANA_PROMPT
import org.junit.Assert.assertFalse
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ChatInputComponentTest {

    @Test
    fun `sana mode should disable send when no image selected even if text exists`() {
        val enabled = ChatInputComponent.shouldEnableSendButton(
            isLoading = false,
            isGenerating = false,
            hasAttachment = false,
            inputText = "hello",
            requiresFaceImage = true
        )
        assertFalse(enabled)
    }

    @Test
    fun `sana mode should enable send when image selected`() {
        val enabled = ChatInputComponent.shouldEnableSendButton(
            isLoading = false,
            isGenerating = false,
            hasAttachment = true,
            inputText = "",
            requiresFaceImage = true
        )
        assertTrue(enabled)
    }

    @Test
    fun `normal mode should enable send when text exists`() {
        val enabled = ChatInputComponent.shouldEnableSendButton(
            isLoading = false,
            isGenerating = false,
            hasAttachment = false,
            inputText = "hello",
            requiresFaceImage = false
        )
        assertTrue(enabled)
    }

    @Test
    fun `sana mode should prefill default prompt when input is blank`() {
        val text = ChatInputComponent.resolveInputTextForModel(
            inputText = "",
            requiresFaceImage = true
        )
        assertEquals(DEFAULT_SANA_PROMPT, text)
    }

    @Test
    fun `sana mode should keep user text when input is not blank`() {
        val text = ChatInputComponent.resolveInputTextForModel(
            inputText = "custom prompt",
            requiresFaceImage = true
        )
        assertEquals("custom prompt", text)
    }
}
