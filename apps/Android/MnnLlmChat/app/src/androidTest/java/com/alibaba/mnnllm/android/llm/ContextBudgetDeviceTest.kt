package com.alibaba.mnnllm.android.llm

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.alibaba.mnnllm.android.chat.chatlist.ChatViewHolders
import com.alibaba.mnnllm.android.chat.model.ChatDataItem
import org.json.JSONObject
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

/**
 * Runs the context budget against a real model and the real native code on the device.
 * Debuggable builds run with CheckJNI, so string/JNI misuse aborts the test process.
 *
 * Needs a text model at [MODEL_DIR]; the tests are skipped when it is missing.
 */
@RunWith(AndroidJUnit4::class)
class ContextBudgetDeviceTest {

    private val sessions = mutableListOf<LlmSession>()

    @Before
    fun requireModel() {
        assumeTrue("model missing at $MODEL_DIR", File(MODEL_DIR, "config.json").exists())
    }

    @After
    fun releaseAll() {
        sessions.forEach { it.release() }
    }

    private fun newSession(
        history: List<ChatDataItem>? = null, summary: String? = null, anchor: Long = -1,
        modelDir: String = MODEL_DIR, limit: Int? = 1024,
    ): LlmSession {
        val session = LlmSession("local/$modelDir", "device-test-${System.nanoTime()}", "$modelDir/config.json",
            history, useCustomConfig = false)
        session.setKeepHistory(true)
        session.contextSummary = summary
        session.contextAnchorId = anchor
        session.load()
        sessions += session
        if (limit != null) {
            session.updateConfig("""{"context_token_limit": $limit, "auto_compress_context": true}""")
        }
        session.updateMaxNewTokens(48)
        return session
    }

    private fun conversation(turns: Int, words: Int, firstUser: String? = null): List<ChatDataItem> =
        (0 until turns).flatMap { t ->
            val user = if (t == 0 && firstUser != null) firstUser else "Turn $t. ${longText(words)}"
            listOf(ChatDataItem(null, ChatViewHolders.USER, user).also { it.dbId = 2L * t + 1 },
                ChatDataItem(null, ChatViewHolders.ASSISTANT, "Answer $t. ${longText(words / 2)}").also { it.dbId = 2L * t + 2 })
        }

    @Test
    fun defaultLimitIsCappedAt8192() {
        assertEquals(8192, newSession(limit = null).contextLimits().limit)
    }

    @Test
    fun longPreExistingConversationCompressesInBoundedTime() {
        // About 60 turns / 15K tokens of history from before the feature existed.
        val session = newSession(conversation(60, 160))
        val check = session.checkContext("Next question")
        assertEquals(ContextAction.COMPRESS, check.action)
        val t0 = System.nanoTime()
        val compressed = session.compressContext()
        val seconds = (System.nanoTime() - t0) / 1e9
        Log.i(TAG, "compression of 60 old turns took $seconds s")
        assertTrue(compressed.success)
        assertTrue("took $seconds s; summarizing must stay bounded", seconds < 60)
        assertTrue(session.countContextTokens() < 1024)
    }

    @Test
    fun stoppingDuringASummaryReturnsPromptly() {
        val session = newSession(conversation(20, 160))
        assertEquals(ContextAction.COMPRESS, session.checkContext("Next").action)
        val done = CountDownLatch(1)
        val worker = Thread { session.compressContext(); done.countDown() }
        worker.start()
        Thread.sleep(1500)  // inside the first summary batch
        val t0 = System.nanoTime()
        session.cancelContextWork()
        assertTrue("compression did not stop", done.await(15, TimeUnit.SECONDS))
        val stopMs = (System.nanoTime() - t0) / 1_000_000
        Log.i(TAG, "summary stopped ${stopMs} ms after cancel")
        assertTrue("stop took $stopMs ms", stopMs < 5000)
        assertFalse(session.ask("still works?")["error"] == true)
    }

    @Test
    fun summaryOfAnImageTurnDoesNotRunTheVisionEncoder() {
        assumeTrue("omni model missing", File(OMNI_DIR, "config.json").exists())
        // The image file does not exist: encoding it would fail or crash.
        val history = conversation(4, 120, firstUser = "<img>/data/local/tmp/does-not-exist.jpg</img>What is in this picture?")
        val session = newSession(history, modelDir = OMNI_DIR)
        assertEquals(ContextAction.COMPRESS, session.checkContext("Summarize our chat. ${longText(120)}").action)
        val compressed = session.compressContext()
        assertTrue(compressed.success)
        assertFalse(compressed.summary.contains("does-not-exist"))
    }

    private fun LlmSession.ask(prompt: String, onChunk: (String) -> Unit = {}): HashMap<String, Any> =
        generate(prompt, emptyMap(), object : GenerateProgressListener {
            override fun onProgress(progress: String?): Boolean {
                progress?.let(onChunk)
                return false
            }
        })

    // Common words, roughly one token each, so word counts approximate token counts.
    private val filler = listOf("apple", "river", "green", "stone", "cloud", "music", "table", "light")
    private fun longText(words: Int) = (0 until words).joinToString(" ") { filler[it % filler.size] }

    private fun thinkingFlag(session: LlmSession): Any? =
        JSONObject(session.dumpConfig()).optJSONObject("jinja")?.optJSONObject("context")?.opt("enable_thinking")

    @Test
    fun decodeStaysInsideTheWindowAndOldTurnsAreTrimmed() {
        val session = newSession()
        var truncated = false
        repeat(8) { turn ->
            val result = session.ask("Turn $turn. ${longText(120)} Reply with one short sentence.")
            assertFalse("turn $turn failed: $result", result["error"] == true)
            val tokens = (result["context_tokens"] as Number).toInt()
            val limit = (result["context_limit"] as Number).toInt()
            assertEquals(1024, limit)
            assertTrue("context $tokens exceeds $limit", tokens <= limit)
            truncated = truncated || result["context_truncated"] == true
            assertTrue((result["peak_anon_kb"] as Number).toLong() > 0)
        }
        assertTrue("eight long turns must trim the 1024-token window", truncated)
    }

    @Test
    fun oversizedMessageIsRejectedWithoutTouchingHistory() {
        val session = newSession()
        session.ask("Hello")
        val before = session.countContextTokens()
        val huge = longText(1500)
        assertEquals(ContextAction.TOO_LONG, session.checkContext(huge).action)
        val result = session.ask(huge)
        assertEquals(true, result["error"])
        assertEquals("context_too_long", result["message"])
        assertEquals(before, session.countContextTokens())
        assertFalse(session.ask("Still fine?")["error"] == true)
    }

    @Test
    fun compressionSummarizesAndRestoresThinkingSwitch() {
        val session = newSession()
        val thinkingBefore = thinkingFlag(session)
        // Build the history with room to spare, then shrink the window so the next prompt
        // lands above the 90% compression trigger while still fitting on its own.
        session.updateConfig("""{"context_token_limit": 4096}""")
        repeat(3) { session.ask("Fact $it: my code word is BLUE$it. ${longText(200)}") }
        session.updateConfig("""{"context_token_limit": 1024}""")
        val prompt = "What are my code words? ${longText(150)}"
        val check = session.checkContext(prompt)
        assertEquals("check=$check", ContextAction.COMPRESS, check.action)
        val tokensBefore = session.countContextTokens()
        val compressed = session.compressContext()
        assertTrue(compressed.success)
        assertTrue(compressed.summary.isNotBlank())
        assertFalse(compressed.summary.contains("<eop>"))
        assertTrue(compressed.keptUserTurns >= 0)
        assertTrue(session.countContextTokens() < tokensBefore)
        assertEquals(thinkingBefore, thinkingFlag(session))
        val result = session.ask(prompt)
        assertFalse(result["error"] == true)
        assertFalse("compression must leave room so kept turns are not trimmed", result["context_truncated"] == true)
    }

    @Test
    fun compressionRestoresAnExistingThinkingSwitchExactly() {
        for (value in listOf(true, false)) {
            val session = newSession()
            session.updateConfig("""{"jinja": {"context": {"enable_thinking": $value}}}""")
            session.updateConfig("""{"context_token_limit": 4096}""")
            repeat(3) { session.ask("Fact $it. ${longText(200)}") }
            session.updateConfig("""{"context_token_limit": 1024}""")
            session.checkContext("Question. ${longText(150)}")
            assertTrue(session.compressContext().success)
            assertEquals(value, thinkingFlag(session))
            session.release()
            sessions.remove(session)
        }
    }

    @Test
    fun emojiAndCjkSurviveTheJniBoundary() {
        val session = newSession()
        val chunks = StringBuilder()
        val result = session.ask("请原样重复：你好😀🚀 then say ok") { chunks.append(it) }
        assertFalse(result["error"] == true)
        assertTrue(chunks.isNotEmpty())
        session.updateSystemPrompt("You are helpful 😀")
        assertTrue(session.getSystemPrompt()!!.contains("😀"))
    }

    @Test
    fun uiThreadCallsNeverWaitForAGeneration() {
        val session = newSession()
        session.updateMaxNewTokens(200)
        val started = CountDownLatch(1)
        val worker = Thread {
            session.ask("Write a long story about the sea. ${longText(50)}") { started.countDown() }
        }
        worker.start()
        assertTrue(started.await(60, TimeUnit.SECONDS))
        val slowest = AtomicLong()
        repeat(50) { i ->
            val t0 = System.nanoTime()
            session.updateConfig("""{"context_token_limit": ${if (i % 2 == 0) 1024 else 2048}}""")
            session.updateSystemPrompt("prompt $i")
            session.updateMaxNewTokens(200)
            session.dumpConfig()
            session.debugInfo
            session.cancelContextWork()
            slowest.set(maxOf(slowest.get(), System.nanoTime() - t0))
        }
        worker.join(120_000)
        Log.i(TAG, "slowest UI-side call batch: ${slowest.get() / 1_000_000} ms")
        assertTrue("UI-side calls blocked for ${slowest.get() / 1_000_000} ms", slowest.get() < 200_000_000)
    }

    @Test
    fun releaseWhileTheUiKeepsCallingDoesNotCrash() {
        val session = newSession()
        session.ask("hi")
        val stop = AtomicBoolean(false)
        val ui = Thread {
            while (!stop.get()) {
                session.updateConfig("""{"auto_compress_context": true}""")
                session.dumpConfig()
                session.getSystemPrompt()
                session.cancelContextWork()
            }
        }
        ui.start()
        Thread.sleep(50)
        session.release()
        sessions.remove(session)
        Thread.sleep(50)
        stop.set(true)
        ui.join(10_000)
        assertEquals("{}", session.dumpConfig())
    }

    @Test
    fun restoredSummaryReplacesHistoryBeforeTheAnchor() {
        fun item(type: Int, id: Long, text: String) = ChatDataItem(null, type, text).also { it.dbId = id }
        val history = listOf(
            item(ChatViewHolders.USER, 1, longText(200)), item(ChatViewHolders.ASSISTANT, 2, longText(200)),
            item(ChatViewHolders.USER, 3, "recent question"), item(ChatViewHolders.ASSISTANT, 4, "recent answer"),
        )
        fun count(summary: String?, anchor: Long): Int {
            val session = newSession(history, summary, anchor)
            val tokens = session.countContextTokens()
            session.release()
            sessions.remove(session)
            return tokens
        }
        val full = count(null, -1)
        val restored = count("The user likes tea.", 3)
        assertTrue("restored $restored should be far below full $full", restored < full / 2)
        // A summary whose anchor row is gone is dropped and the full history is used.
        assertEquals(full, count("stale", 99))
    }

    companion object {
        private const val TAG = "ContextBudgetDeviceTest"
        private const val MODEL_DIR = "/data/local/tmp/mnn_models/Qwen2.5-0.5B-Instruct-MNN"
        private const val OMNI_DIR = "/data/local/tmp/mnn_models/gemma-4-E2B-it-MNN"
    }
}
