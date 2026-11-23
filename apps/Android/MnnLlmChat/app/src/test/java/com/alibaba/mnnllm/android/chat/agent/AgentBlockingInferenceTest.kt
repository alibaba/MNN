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
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

@OptIn(ExperimentalCoroutinesApi::class)
class AgentBlockingInferenceTest {
    private val request = AgentInferenceRequest(listOf(AgentMessage(AgentRole.USER, "hello")), false, false)

    @Test
    fun `blocking submit streams native-compatible history and returns metrics`() = runTest {
        val deltas = mutableListOf<String>()
        val inference = AgentBlockingInference(submit = { input, progress ->
            assertEquals(listOf("user" to "hello"), input.nativeHistory())
            assertFalse(progress("hel"))
            assertFalse(progress("lo"))
            progress(null)
            mapOf("success" to true, "decode_len" to 2)
        }, cancel = { }, dispatcher = StandardTestDispatcher(testScheduler))
        val generation = inference.generate(request, deltas::add)
        assertEquals(listOf("hel", "lo"), deltas)
        assertEquals("hello", generation.text)
        assertEquals(2, generation.metrics["decode_len"])
    }

    @Test
    fun `cancellation invokes native cancel hook before awaiting blocking work completion`() = runTest {
        val entered = CompletableDeferred<Unit>()
        val stopped = AtomicBoolean(false)
        val nativeWait = CountDownLatch(1)
        val inference = AgentBlockingInference(submit = { _, progress ->
            entered.complete(Unit)
            check(nativeWait.await(5, TimeUnit.SECONDS)) { "Native cancel hook did not run" }
            assertTrue(progress("Must not be delivered"))
            mapOf("success" to true)
        }, cancel = { stopped.set(true); nativeWait.countDown() })
        var delivered = false
        val job = launch { inference.generate(request) { delivered = true } }
        entered.await()
        job.cancel()
        job.join()
        assertTrue(stopped.get())
        assertFalse(delivered)
    }

    @Test
    fun `native failure and oversized model output are not mistaken for final answers`() = runTest {
        val badResult = AgentBlockingInference(submit = { _, _ -> mapOf("error" to true) }, cancel = { },
            dispatcher = StandardTestDispatcher(testScheduler))
        try {
            badResult.generate(request) { }
            throw AssertionError("Native failure was ignored")
        } catch (_: IllegalStateException) { }
        val overflow = AgentBlockingInference(submit = { _, progress ->
            assertTrue(progress("x".repeat(AgentOutputParser.MAX_OUTPUT_LENGTH + 1)))
            mapOf("success" to true)
        }, cancel = { }, dispatcher = StandardTestDispatcher(testScheduler))
        try {
            overflow.generate(request) { }
            throw AssertionError("Output limit was ignored")
        } catch (_: IllegalArgumentException) { }
    }
}
