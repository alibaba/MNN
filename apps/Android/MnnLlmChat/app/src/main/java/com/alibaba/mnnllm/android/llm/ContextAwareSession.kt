package com.alibaba.mnnllm.android.llm

/** Outcome of the pre-send context budget check; codes match native mls::ContextAction. */
enum class ContextAction(val code: Int) {
    SEND(0),
    COMPRESS(1),
    TOO_LONG(2);

    companion object {
        fun fromCode(code: Int): ContextAction = entries.firstOrNull { it.code == code } ?: SEND
    }
}

data class ContextCheck(
    val tokens: Int,
    val budget: Int,
    val limit: Int,
    val action: ContextAction,
) {
    companion object {
        val NONE = ContextCheck(0, 0, 0, ContextAction.SEND)
    }
}

data class ContextCompression(
    val success: Boolean,
    val summary: String,
    /** User turns still kept verbatim after the summary; used to anchor the summary. */
    val keptUserTurns: Int,
)

data class ContextLimits(val limit: Int, val modelMax: Int)

/**
 * A chat session that keeps its conversation inside a token budget.
 *
 * Everything except [cancelContextWork] may block for a long time (tokenizing the history,
 * generating a summary) and must be called from a worker thread.
 */
interface ContextAwareSession {
    /**
     * Summary of the turns before the stored row [contextAnchorId]. Applied at
     * [ChatSession.load]: only history from the anchor row on reaches the model, the
     * rest is represented by the summary. The full history stays available to the UI.
     */
    var contextSummary: String?
    var contextAnchorId: Long

    fun checkContext(prompt: String): ContextCheck

    fun compressContext(): ContextCompression

    fun countContextTokens(): Int

    fun contextLimits(): ContextLimits

    /** Stops a running [compressContext] after its current batch. Safe on any thread. */
    fun cancelContextWork()
}
