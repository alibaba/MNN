package com.alibaba.mnnllm.android.chat

import com.alibaba.mnnllm.android.chat.chatlist.ChatViewHolders
import com.alibaba.mnnllm.android.chat.model.ChatDataItem
import com.alibaba.mnnllm.android.chat.model.StoredContextSummary
import com.alibaba.mnnllm.android.llm.ContextAction
import com.alibaba.mnnllm.android.llm.ContextAwareSession
import com.alibaba.mnnllm.android.llm.ContextCheck
import com.alibaba.mnnllm.android.llm.ContextCompression
import com.alibaba.mnnllm.android.llm.ContextLimits
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.UnconfinedTestDispatcher
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class ChatContextManagerTest {

    private class FakeStore(
        var stored: StoredContextSummary? = null,
        var userIds: List<Long> = emptyList(),
    ) : ContextSummaryStore {
        var cleared = false
        override fun load(sessionId: String) = stored
        override fun save(sessionId: String, summary: String, anchorId: Long) {
            stored = StoredContextSummary(summary, anchorId)
        }
        override fun clear(sessionId: String) {
            stored = null
            cleared = true
        }
        override fun userMessageIds(sessionId: String) = userIds
    }

    private class FakeSession(
        var compression: ContextCompression = ContextCompression(false, "", 0),
        var throwOnCompress: Boolean = false,
    ) : ContextAwareSession {
        override var contextSummary: String? = null
        override var contextAnchorId: Long = -1L
        var compressCalls = 0
        override fun checkContext(prompt: String) = ContextCheck.NONE
        override fun compressContext(): ContextCompression {
            compressCalls++
            if (throwOnCompress) throw IllegalStateException("boom")
            return compression
        }
        override fun countContextTokens() = 321
        override fun contextLimits() = ContextLimits(4096, 32768)
        override fun cancelContextWork() {}
    }

    private val compressCheck = ContextCheck(3800, 3072, 4096, ContextAction.COMPRESS)

    private fun item(type: Int, dbId: Long, text: String = "t$dbId") =
        ChatDataItem(null, type, text).also { it.dbId = dbId }

    @Test
    fun `anchor is the first kept user row, ignoring the message being sent`() {
        val ids = listOf(10L, 12L, 14L, 16L, 18L)  // 18 is the pending message
        assertEquals(14L, ChatContextManager.anchorFor(ids, 2, pendingUserRowId = 18L))
        assertEquals(16L, ChatContextManager.anchorFor(ids, 1, pendingUserRowId = 18L))
        assertEquals(10L, ChatContextManager.anchorFor(ids, 4, pendingUserRowId = 18L))
        assertNull(ChatContextManager.anchorFor(ids, 5, pendingUserRowId = 18L))
        // No turn kept verbatim: the summary covers everything before the pending message.
        assertEquals(18L, ChatContextManager.anchorFor(ids, 0, pendingUserRowId = 18L))
        assertNull(ChatContextManager.anchorFor(ids, 0, pendingUserRowId = -1L))
        assertNull(ChatContextManager.anchorFor(ids, -1, pendingUserRowId = 18L))
        assertNull(ChatContextManager.anchorFor(emptyList(), 1))
    }

    @Test
    fun `anchor when the pending message could not be stored`() {
        // Nothing to exclude: every stored user row is part of native history.
        assertEquals(16L, ChatContextManager.anchorFor(listOf(10L, 12L, 14L, 16L), 1, pendingUserRowId = -1L))
    }

    @Test
    fun `restore returns the summary when its anchor is in the history`() {
        val store = FakeStore(StoredContextSummary("S", 3L))
        val history = listOf(item(ChatViewHolders.USER, 1), item(ChatViewHolders.ASSISTANT, 2),
            item(ChatViewHolders.USER, 3), item(ChatViewHolders.ASSISTANT, 4))
        val restored = ChatContextManager(store).restore("s", history)
        assertEquals(StoredContextSummary("S", 3L), restored)
        assertFalse(store.cleared)
    }

    @Test
    fun `restore drops a summary whose anchor row is gone`() {
        val store = FakeStore(StoredContextSummary("S", 99L))
        val restored = ChatContextManager(store).restore("s", listOf(item(ChatViewHolders.USER, 1)))
        assertNull(restored)
        assertTrue(store.cleared)
    }

    @Test
    fun `restore without session or history is a no-op`() {
        val store = FakeStore(StoredContextSummary("S", 1L))
        val manager = ChatContextManager(store)
        assertNull(manager.restore(null, listOf(item(ChatViewHolders.USER, 1))))
        assertNull(manager.restore("s", emptyList()))
        assertFalse(store.cleared)
    }

    @Test
    fun `apply hands summary and anchor to the session`() {
        val session = FakeSession()
        val manager = ChatContextManager(FakeStore())
        manager.apply(session, StoredContextSummary("S", 7L))
        assertEquals("S", session.contextSummary)
        assertEquals(7L, session.contextAnchorId)
        manager.apply(session, null)
        assertNull(session.contextSummary)
        assertEquals(-1L, session.contextAnchorId)
    }

    @Test
    fun `successful compression is stored with its anchor`() {
        val store = FakeStore(userIds = listOf(1L, 3L, 5L, 7L, 9L))
        val session = FakeSession(ContextCompression(true, "summary", 2))
        val manager = ChatContextManager(store)
        manager.compressIfNeeded(session, "s", compressCheck, pendingUserRowId = 9L)
        assertNull("stored only once the turn has run", store.stored)
        assertFalse(manager.usage.value.compressing)
        manager.onTurnFinished(hashMapOf("context_limit" to 4096L, "context_tokens" to 900L), pendingUserRowId = 9L)
        assertEquals(StoredContextSummary("summary", 5L), store.stored)
    }

    @Test
    fun `summary anchored on a rejected message moves to the next message that runs`() {
        val store = FakeStore(userIds = listOf(1L, 3L, 5L))
        val manager = ChatContextManager(store)
        manager.compressIfNeeded(FakeSession(ContextCompression(true, "S", 0)), "s", compressCheck, pendingUserRowId = 5L)
        // Native rejects row 5 after prefill; the presenter deletes it.
        manager.onTurnFinished(hashMapOf("error" to true, "message" to ChatPresenter.CONTEXT_TOO_LONG), pendingUserRowId = 5L)
        assertNull(store.stored)
        // A later rejection keeps waiting; the first message that runs becomes the anchor.
        manager.onTurnFinished(hashMapOf("error" to true, "message" to ChatPresenter.CONTEXT_TOO_LONG), pendingUserRowId = 7L)
        assertNull(store.stored)
        manager.onTurnFinished(hashMapOf("context_limit" to 4096L), pendingUserRowId = 9L)
        assertEquals(StoredContextSummary("S", 9L), store.stored)
    }

    @Test
    fun `summary with a kept-turn anchor survives a rejection of the current message`() {
        val store = FakeStore(userIds = listOf(1L, 3L, 5L, 7L))
        val manager = ChatContextManager(store)
        manager.compressIfNeeded(FakeSession(ContextCompression(true, "S", 1)), "s", compressCheck, pendingUserRowId = 7L)
        manager.onTurnFinished(hashMapOf("error" to true, "message" to ChatPresenter.CONTEXT_TOO_LONG), pendingUserRowId = 7L)
        assertEquals(StoredContextSummary("S", 5L), store.stored)
    }

    @Test
    fun `reset drops a summary that was never stored`() {
        val store = FakeStore(userIds = listOf(1L, 3L))
        val manager = ChatContextManager(store)
        manager.compressIfNeeded(FakeSession(ContextCompression(true, "S", 0)), "s", compressCheck, pendingUserRowId = 3L)
        manager.onConversationReset()
        manager.onTurnFinished(hashMapOf("context_limit" to 4096L), pendingUserRowId = 11L)
        assertNull(store.stored)
    }

    @Test
    fun `compression runs only when the check asks for it`() {
        val session = FakeSession(ContextCompression(true, "summary", 1))
        val manager = ChatContextManager(FakeStore(userIds = listOf(1L, 2L)))
        manager.compressIfNeeded(session, "s", compressCheck.copy(action = ContextAction.SEND))
        manager.compressIfNeeded(session, null, compressCheck)
        assertEquals(0, session.compressCalls)
    }

    @Test
    fun `a failed or throwing compression keeps the stored summary and clears the flag`() {
        val old = StoredContextSummary("old", 1L)
        val store = FakeStore(old, listOf(1L, 2L, 3L))
        val manager = ChatContextManager(store)
        manager.compressIfNeeded(FakeSession(ContextCompression(false, "", 0)), "s", compressCheck)
        manager.compressIfNeeded(FakeSession(throwOnCompress = true), "s", compressCheck)
        assertEquals(old, store.stored)
        assertFalse(manager.usage.value.compressing)
    }

    @Test
    fun `compression without a matching anchor is not stored`() {
        val store = FakeStore(userIds = listOf(1L))  // only the pending message
        ChatContextManager(store).compressIfNeeded(FakeSession(ContextCompression(true, "s", 2)), "s", compressCheck, pendingUserRowId = 1L)
        assertNull(store.stored)
    }

    @OptIn(ExperimentalCoroutinesApi::class)
    @Test
    fun `truncation notice is shown once per conversation`() = runTest {
        val manager = ChatContextManager(FakeStore())
        val notices = mutableListOf<ContextNotice>()
        val job = launch(UnconfinedTestDispatcher(testScheduler)) { manager.notices.toList(notices) }
        val truncated = hashMapOf<String, Any>("context_tokens" to 900L, "context_limit" to 4096L, "context_truncated" to true)
        manager.onTurnFinished(truncated)
        manager.onTurnFinished(truncated)
        assertEquals(listOf(ContextNotice.TRUNCATED), notices)
        manager.onConversationReset()
        manager.onTurnFinished(truncated)
        assertEquals(listOf(ContextNotice.TRUNCATED, ContextNotice.TRUNCATED), notices)
        job.cancel()
    }

    @Test
    fun `turn telemetry and refresh update the usage state`() {
        val manager = ChatContextManager(FakeStore())
        manager.onTurnFinished(hashMapOf("context_tokens" to 1234L, "context_limit" to 4096L))
        assertEquals(1234, manager.usage.value.tokens)
        assertEquals(4096, manager.usage.value.limit)
        manager.onTurnFinished(hashMapOf("response" to ""))  // no telemetry: unchanged
        assertEquals(1234, manager.usage.value.tokens)
        manager.refresh(FakeSession())
        assertEquals(321, manager.usage.value.tokens)
        assertEquals(32768, manager.usage.value.modelMax)
        manager.onConversationReset()
        assertEquals(0, manager.usage.value.tokens)
        assertEquals(4096, manager.usage.value.limit)
    }

    @Test
    fun `default onContextRejected finishes the request so voice chat does not hang`() {
        var finished: Map<String, Any>? = null
        val listener = object : ChatPresenter.GenerateListener {
            override fun onDiffusionGenerateProgress(progress: String?, diffusionDestPath: String?) {}
            override fun onGenerateStart() {}
            override fun onGenerateFinished(benchMarkResult: HashMap<String, Any>) { finished = benchMarkResult }
            override fun onLlmGenerateProgress(progress: String?, generateResultProcessor: GenerateResultProcessor) {}
        }
        listener.onContextRejected(item(ChatViewHolders.USER, -1))
        assertEquals(true, finished?.get("error"))
        assertEquals(ChatPresenter.CONTEXT_TOO_LONG, finished?.get("message"))
    }

    @Test
    fun `usage state flags`() {
        assertFalse(ContextUsageState().visible)
        assertTrue(ContextUsageState(tokens = 3300, limit = 4096).nearlyFull)
        assertFalse(ContextUsageState(tokens = 3000, limit = 4096).nearlyFull)
    }
}
