package com.alibaba.mnnllm.android.chat

import android.util.Log
import com.alibaba.mnnllm.android.chat.model.ChatDataItem
import com.alibaba.mnnllm.android.chat.model.ChatDataManager
import com.alibaba.mnnllm.android.chat.model.StoredContextSummary
import com.alibaba.mnnllm.android.llm.ContextAction
import com.alibaba.mnnllm.android.llm.ContextAwareSession
import com.alibaba.mnnllm.android.llm.ContextCheck
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update

/** What the context usage indicator shows. */
data class ContextUsageState(
    val tokens: Int = 0,
    val limit: Int = 0,
    val modelMax: Int = 0,
    val compressing: Boolean = false,
) {
    val visible: Boolean get() = limit > 0
    val nearlyFull: Boolean get() = limit > 0 && tokens * 5 >= limit * 4
}

enum class ContextNotice { TRUNCATED, COMPRESSED }

/** Persistence the manager needs; [ChatDataManager] in the app, a fake in tests. */
interface ContextSummaryStore {
    fun load(sessionId: String): StoredContextSummary?
    fun save(sessionId: String, summary: String, anchorId: Long)
    fun clear(sessionId: String)
    fun userMessageIds(sessionId: String): List<Long>
}

class ChatDataContextStore(private val chatDataManager: ChatDataManager) : ContextSummaryStore {
    override fun load(sessionId: String) = chatDataManager.getContextSummary(sessionId)
    override fun save(sessionId: String, summary: String, anchorId: Long) =
        chatDataManager.setContextSummary(sessionId, summary, anchorId)
    override fun clear(sessionId: String) = chatDataManager.setContextSummary(sessionId, null, -1L)
    override fun userMessageIds(sessionId: String) = chatDataManager.getUserMessageIds(sessionId)
}

/**
 * Owns the conversation's context budget on the Kotlin side: restoring a stored summary
 * when a session is opened, the pre-send check, summarizing older turns, persisting the
 * summary and the state of the usage indicator.
 *
 * The summary is anchored on the database row of the first user message it does NOT
 * cover. Counting user rows (rather than messages) keeps the anchor right when error
 * replies were stored or an empty reply was skipped.
 *
 * Blocking methods are documented as such and must run on a worker thread.
 */
class ChatContextManager(private val store: ContextSummaryStore) {

    private val _usage = MutableStateFlow(ContextUsageState())
    val usage: StateFlow<ContextUsageState> = _usage.asStateFlow()

    private val _notices = MutableSharedFlow<ContextNotice>(extraBufferCapacity = 4)
    val notices: SharedFlow<ContextNotice> = _notices

    @Volatile
    private var truncationNoticeShown = false

    /**
     * The stored summary for [sessionId] if its anchor row is part of [history]. When the
     * anchor is gone the summary no longer lines up with the history, so it is deleted and
     * the full history is used (native trimming still bounds it). Blocking.
     */
    fun restore(sessionId: String?, history: List<ChatDataItem>?): StoredContextSummary? {
        if (sessionId.isNullOrEmpty() || history.isNullOrEmpty()) {
            return null
        }
        val stored = store.load(sessionId) ?: return null
        if (history.none { it.dbId == stored.anchorId }) {
            Log.w(TAG, "Summary anchor ${stored.anchorId} not found in session $sessionId; dropping summary")
            store.clear(sessionId)
            return null
        }
        return stored
    }

    /** Hands a restored summary to a session that has not been loaded yet. */
    fun apply(session: ContextAwareSession, restored: StoredContextSummary?) {
        session.contextSummary = restored?.summary
        session.contextAnchorId = restored?.anchorId ?: -1L
    }

    /** Pre-send check; does not change the conversation. Blocking. */
    fun check(session: ContextAwareSession, prompt: String): ContextCheck {
        return try {
            session.checkContext(prompt)
        } catch (e: Exception) {
            // The check is advisory: native still enforces the budget during Response.
            Log.e(TAG, "Context check failed", e)
            ContextCheck.NONE
        }
    }

    /**
     * Summarizes older turns when [check] asks for it. The current user message must already
     * be stored, so it is excluded when locating the anchor. Failures only cost the summary:
     * native trimming still keeps the prompt within the budget. Blocking.
     */
    fun compressIfNeeded(
        session: ContextAwareSession,
        sessionId: String?,
        check: ContextCheck,
        pendingUserRowId: Long = -1L,
    ) {
        if (check.action != ContextAction.COMPRESS || sessionId.isNullOrEmpty()) {
            return
        }
        _usage.update { it.copy(compressing = true) }
        try {
            val result = session.compressContext()
            if (!result.success || result.summary.isEmpty()) {
                return
            }
            val anchor = anchorFor(store.userMessageIds(sessionId), result.keptUserTurns, pendingUserRowId)
            if (anchor != null) {
                store.save(sessionId, result.summary, anchor)
            } else {
                // Native keeps the summary for this session; it just won't survive a reload.
                Log.w(TAG, "No anchor for ${result.keptUserTurns} kept turns in session $sessionId")
            }
            _notices.tryEmit(ContextNotice.COMPRESSED)
        } catch (e: Exception) {
            Log.e(TAG, "Context compression failed", e)
        } finally {
            _usage.update { it.copy(compressing = false) }
        }
    }

    /** Reads the context telemetry of a finished turn. */
    fun onTurnFinished(result: Map<String, Any>) {
        val limit = (result["context_limit"] as? Number)?.toInt() ?: return
        val tokens = (result["context_tokens"] as? Number)?.toInt()
        _usage.update { it.copy(tokens = tokens ?: it.tokens, limit = limit, compressing = false) }
        if (result["context_truncated"] == true && !truncationNoticeShown) {
            truncationNoticeShown = true
            _notices.tryEmit(ContextNotice.TRUNCATED)
        }
    }

    /** Recomputes the indicator from native state, e.g. after loading or a settings change. Blocking. */
    fun refresh(session: ContextAwareSession) {
        try {
            val limits = session.contextLimits()
            val tokens = session.countContextTokens()
            _usage.update { it.copy(tokens = tokens, limit = limits.limit, modelMax = limits.modelMax) }
        } catch (e: Exception) {
            Log.e(TAG, "Context refresh failed", e)
        }
    }

    /** A new conversation starts empty and may warn about trimming again. */
    fun onConversationReset() {
        truncationNoticeShown = false
        _usage.update { it.copy(tokens = 0, compressing = false) }
    }

    companion object {
        private const val TAG = "ChatContextManager"

        /**
         * Row id of the first user message still kept verbatim. [userIds] are the session's
         * user rows oldest first. The message being sent ([pendingUserRowId], -1 if it was
         * not stored) is not in native history yet, so it is not one of the kept turns.
         */
        fun anchorFor(userIds: List<Long>, keptUserTurns: Int, pendingUserRowId: Long = -1L): Long? {
            if (keptUserTurns == 0) {
                // Everything before the message being sent is in the summary.
                return pendingUserRowId.takeIf { it >= 0 && it in userIds }
            }
            val consumed = if (pendingUserRowId >= 0) userIds.filter { it != pendingUserRowId } else userIds
            if (keptUserTurns < 0 || keptUserTurns > consumed.size) {
                return null
            }
            return consumed[consumed.size - keptUserTurns]
        }
    }
}
