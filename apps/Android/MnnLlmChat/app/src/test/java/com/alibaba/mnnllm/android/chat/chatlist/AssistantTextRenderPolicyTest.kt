package com.alibaba.mnnllm.android.chat.chatlist

import com.alibaba.mnnllm.android.chat.model.ChatDataItem
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class AssistantTextRenderPolicyTest {

    @Test
    fun `uses plain text when waiting hint is forced`() {
        val item = ChatDataItem(ChatViewHolders.ASSISTANT).apply {
            forceShowLoadingWithText = true
        }
        assertTrue(AssistantTextRenderPolicy.usePlainText(item))
    }

    @Test
    fun `uses markdown for normal assistant responses`() {
        val item = ChatDataItem(ChatViewHolders.ASSISTANT).apply {
            forceShowLoadingWithText = false
        }
        assertFalse(AssistantTextRenderPolicy.usePlainText(item))
    }
}
