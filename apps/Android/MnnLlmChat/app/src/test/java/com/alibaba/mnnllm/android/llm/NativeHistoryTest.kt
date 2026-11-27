package com.alibaba.mnnllm.android.llm

import com.alibaba.mnnllm.android.chat.chatlist.ChatViewHolders
import com.alibaba.mnnllm.android.chat.model.ChatDataItem
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class NativeHistoryTest {
    private fun user(text: String?) = ChatDataItem(null, ChatViewHolders.USER, text)
    private fun bot(text: String?) = ChatDataItem(null, ChatViewHolders.ASSISTANT, text)

    @Test
    fun `regular pairs pass through`() {
        assertEquals(listOf("u1", "a1", "u2", "a2"), NativeHistory.alternate(listOf(user("u1"), bot("a1"), user("u2"), bot("a2"))))
    }

    @Test
    fun `a user turn without a reply gets an empty reply`() {
        assertEquals(listOf("u1", "", "u2", "a2"), NativeHistory.alternate(listOf(user("u1"), user("u2"), bot("a2"))))
    }

    @Test
    fun `leading replies, headers and null text are skipped`() {
        val items = listOf(ChatDataItem(null, ChatViewHolders.HEADER, "h"), bot("orphan"), user(null), user("u1"), bot("a1"), bot("dup"))
        assertEquals(listOf("u1", "a1"), NativeHistory.alternate(items))
    }

    @Test
    fun `a trailing user turn stays unanswered`() {
        assertEquals(listOf("u1"), NativeHistory.alternate(listOf(user("u1"))))
    }
}
