package com.alibaba.mnnllm.android.chat.chatlist

import androidx.recyclerview.widget.RecyclerView
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ChatListAutoScrollPolicyTest {

    @Test
    fun `programmatic settling should not mark user scrolling`() {
        val next = ChatListComponent.resolveUserScrollingState(
            currentUserScrolling = false,
            newState = RecyclerView.SCROLL_STATE_SETTLING,
            isAtBottom = true
        )
        assertFalse(next)
    }

    @Test
    fun `dragging should mark user scrolling`() {
        val next = ChatListComponent.resolveUserScrollingState(
            currentUserScrolling = false,
            newState = RecyclerView.SCROLL_STATE_DRAGGING,
            isAtBottom = false
        )
        assertTrue(next)
    }

    @Test
    fun `idle at bottom should clear user scrolling`() {
        val next = ChatListComponent.resolveUserScrollingState(
            currentUserScrolling = true,
            newState = RecyclerView.SCROLL_STATE_IDLE,
            isAtBottom = true
        )
        assertFalse(next)
    }
}

