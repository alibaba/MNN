package com.alibaba.mnnllm.android.chat.agent

import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Collections
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

@OptIn(ExperimentalCoroutinesApi::class)
class LlmSessionAgentInferenceTest {
    private data class Settings(val tokens: Int, val audio: Boolean, val thinking: Boolean)

    private class FakeSession : AgentSessionAccess {
        val original = Settings(256, true, true)
        var settings = original
        val events: MutableList<String> = Collections.synchronizedList(mutableListOf())
        var captures = 0
        var configurations = 0
        var restorations = 0
        var throwOnConfigure = false
        var onCancel: () -> Unit = {}
        var onSubmit: (AgentInferenceRequest, (String?) -> Boolean) -> Map<String, Any> = { _, progress ->
            assertFalse(progress("6"))
            mapOf("success" to true)
        }

        override fun captureConfiguration(): AgentPassConfiguration {
            captures++
            events.add("capture")
            val before = settings
            return object : AgentPassConfiguration {
                override fun configure(request: AgentInferenceRequest) {
                    configurations++
                    events.add("configure")
                    settings = Settings(request.outputTokenLimit, false, request.allowThinking)
                    if (throwOnConfigure) throw IllegalArgumentException("Invalid configuration")
                }
                override fun restore() {
                    restorations++
                    events.add("restore")
                    settings = before
                }
            }
        }

        override fun submit(request: AgentInferenceRequest, onChunk: (String?) -> Boolean): Map<String, Any> {
            events.add("submit_start")
            assertEquals(Settings(request.outputTokenLimit, false, request.allowThinking), settings)
            return try { onSubmit(request, onChunk) } finally { events.add("submit_end") }
        }

        override fun cancel() { events.add("cancel"); onCancel() }
    }

    private val request = AgentInferenceRequest(listOf(AgentMessage(AgentRole.USER, "2*3")), true, false, 32)

    @Test
    fun `default adapter applies native token thinking and audio settings then restores`() = runTest {
        val session = FakeSession()
        val adapter = LlmSessionAgentInference(session, dispatcher = StandardTestDispatcher(testScheduler))
        assertEquals("6", adapter.generate(request) {}.text)
        assertEquals(session.original, session.settings)
        assertEquals(1, session.captures)
        assertEquals(1, session.configurations)
        assertEquals(1, session.restorations)
        assertEquals(listOf("capture", "configure", "submit_start", "submit_end", "restore"), session.events.toList())
    }

    @Test
    fun `default adapter restores partial configuration even when preparation fails`() = runTest {
        val session = FakeSession().also { it.throwOnConfigure = true }
        val adapter = LlmSessionAgentInference(session, dispatcher = StandardTestDispatcher(testScheduler))
        try {
            adapter.generate(request) {}
            throw AssertionError("Preparation failure was ignored")
        } catch (_: IllegalArgumentException) {}
        assertEquals(session.original, session.settings)
        assertEquals(listOf("capture", "configure", "restore"), session.events.toList())
    }

    @Test
    fun `default adapter restores after native exceptions and failed result maps`() = runTest {
        for (throwOnSubmit in listOf(true, false)) {
            val session = FakeSession().also {
                it.onSubmit = { _, _ ->
                    if (throwOnSubmit) throw IllegalStateException("Native failure")
                    mapOf("error" to true)
                }
            }
            val adapter = LlmSessionAgentInference(session, dispatcher = StandardTestDispatcher(testScheduler))
            try {
                adapter.generate(request) {}
                throw AssertionError("Native failure was ignored")
            } catch (_: IllegalStateException) {}
            assertEquals(session.original, session.settings)
            assertEquals(1, session.restorations)
            assertTrue(session.events.indexOf("restore") > session.events.indexOf("submit_end"))
        }
    }

    @Test
    fun `cancellation restores only after blocking native submit has actually exited`() = runTest {
        val session = FakeSession()
        val entered = CompletableDeferred<Unit>()
        val nativeWait = CountDownLatch(1)
        val nativeRunning = AtomicBoolean(false)
        session.onCancel = { nativeWait.countDown() }
        session.onSubmit = { _, progress ->
            nativeRunning.set(true)
            entered.complete(Unit)
            try {
                check(nativeWait.await(5, TimeUnit.SECONDS)) { "Cancel hook did not run" }
                assertTrue(progress("Must not be displayed"))
                mapOf("success" to true)
            } finally { nativeRunning.set(false) }
        }
        val adapter = LlmSessionAgentInference(session)
        var displayed = false
        val job = launch { adapter.generate(request) { displayed = true } }
        entered.await()
        job.cancel()
        job.join()
        assertTrue(job.isCancelled)
        assertFalse(nativeRunning.get())
        assertFalse(displayed)
        assertEquals(session.original, session.settings)
        assertEquals(1, session.restorations)
        assertTrue(session.events.indexOf("restore") > session.events.indexOf("submit_end"))
    }

    @Test
    fun `explicit UI preparation keeps configuration ownership with the caller`() = runTest {
        val session = FakeSession()
        val callerSnapshot = session.captureConfiguration()
        val adapter = LlmSessionAgentInference(session, preparePass = { callerSnapshot.configure(it) },
            dispatcher = StandardTestDispatcher(testScheduler))
        try {
            assertEquals("6", adapter.generate(request) {}.text)
            assertEquals(1, session.captures)
            assertEquals(1, session.configurations)
            assertEquals(0, session.restorations)
            assertEquals(Settings(32, false, false), session.settings)
        } finally { callerSnapshot.restore() }
        assertEquals(1, session.restorations)
        assertEquals(session.original, session.settings)
    }

    @Test
    fun `cancellation during caller preparation prevents a subsequent native submit`() = runTest {
        val session = FakeSession()
        val callerSnapshot = session.captureConfiguration()
        val entered = CompletableDeferred<Unit>()
        val preparationWait = CountDownLatch(1)
        session.onCancel = { preparationWait.countDown() }
        val adapter = LlmSessionAgentInference(session, preparePass = {
            callerSnapshot.configure(it)
            entered.complete(Unit)
            check(preparationWait.await(5, TimeUnit.SECONDS)) { "Cancel hook did not run" }
        })
        val job = launch { adapter.generate(request) {} }
        try {
            entered.await()
            job.cancel()
            job.join()
            assertTrue(job.isCancelled)
            assertFalse(session.events.contains("submit_start"))
            assertEquals(0, session.restorations)
        } finally { callerSnapshot.restore() }
        assertEquals(session.original, session.settings)
    }
}
