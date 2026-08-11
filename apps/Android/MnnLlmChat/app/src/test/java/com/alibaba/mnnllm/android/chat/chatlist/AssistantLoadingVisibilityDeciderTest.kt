package com.alibaba.mnnllm.android.chat.chatlist

import com.alibaba.mnnllm.android.chat.model.ChatDataItem
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class AssistantLoadingVisibilityDeciderTest {

    @Test
    fun `shows loading when forcing loading animation with text during generation`() {
        val item = ChatDataItem(ChatViewHolders.ASSISTANT).apply {
            loading = true
            hasOmniAudio = false
            displayText = "OpenCL 生成图片约需 1 分钟，请耐心等待"
            forceShowLoadingWithText = true
        }

        assertTrue(AssistantLoadingVisibilityDecider.shouldShow(item))
    }

    @Test
    fun `hides loading when text exists and no force flag for normal assistant response`() {
        val item = ChatDataItem(ChatViewHolders.ASSISTANT).apply {
            loading = true
            hasOmniAudio = false
            displayText = "normal response"
            forceShowLoadingWithText = false
        }

        assertFalse(AssistantLoadingVisibilityDecider.shouldShow(item))
    }

    @Test
    fun `omni audio loading still depends on loading flag`() {
        val item = ChatDataItem(ChatViewHolders.ASSISTANT).apply {
            hasOmniAudio = true
            displayText = "any text"
            forceShowLoadingWithText = true
        }

        item.loading = true
        assertTrue(AssistantLoadingVisibilityDecider.shouldShow(item))

        item.loading = false
        assertFalse(AssistantLoadingVisibilityDecider.shouldShow(item))
    }
}
