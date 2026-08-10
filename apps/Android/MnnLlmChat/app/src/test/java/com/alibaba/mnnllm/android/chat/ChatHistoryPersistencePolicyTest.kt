package com.alibaba.mnnllm.android.chat

import com.alibaba.mnnllm.android.chat.chatlist.ChatViewHolders
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ChatHistoryPersistencePolicyTest {

    @Test
    fun `saves partial assistant response when llm generation is interrupted`() {
        assertTrue(
            ChatHistoryPersistencePolicy.shouldSaveInterruptedAssistant(
                isGenerating = true,
                isMockStreamSession = false,
                isDiffusion = false,
                itemType = ChatViewHolders.ASSISTANT,
                text = "partial response"
            )
        )
    }

    @Test
    fun `does not save an empty assistant placeholder`() {
        assertFalse(
            ChatHistoryPersistencePolicy.shouldSaveInterruptedAssistant(
                isGenerating = true,
                isMockStreamSession = false,
                isDiffusion = false,
                itemType = ChatViewHolders.ASSISTANT,
                text = ""
            )
        )
    }

    @Test
    fun `does not save diffusion progress text`() {
        assertFalse(
            ChatHistoryPersistencePolicy.shouldSaveInterruptedAssistant(
                isGenerating = true,
                isMockStreamSession = false,
                isDiffusion = true,
                itemType = ChatViewHolders.ASSISTANT,
                text = "Generating image"
            )
        )
    }

    @Test
    fun `does not save when generation already finished`() {
        assertFalse(
            ChatHistoryPersistencePolicy.shouldSaveInterruptedAssistant(
                isGenerating = false,
                isMockStreamSession = false,
                isDiffusion = false,
                itemType = ChatViewHolders.ASSISTANT,
                text = "completed response"
            )
        )
    }

    @Test
    fun `does not save debug mock stream output`() {
        assertFalse(
            ChatHistoryPersistencePolicy.shouldSaveInterruptedAssistant(
                isGenerating = true,
                isMockStreamSession = true,
                isDiffusion = false,
                itemType = ChatViewHolders.ASSISTANT,
                text = "mock response"
            )
        )
    }
}
