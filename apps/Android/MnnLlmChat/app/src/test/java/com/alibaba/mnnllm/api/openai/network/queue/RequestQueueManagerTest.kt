package com.alibaba.mnnllm.api.openai.network.queue

import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.async
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class RequestQueueManagerTest {
    @Test
    fun statsDoNotReorderRequestsAndClearCompletesWaiters() = runBlocking {
        withTimeout(5000) {
            val queue = RequestQueueManager.getInstance()
            queue.clearQueue()
            val started = CompletableDeferred<Unit>()
            val gate = CompletableDeferred<Unit>()
            val order = mutableListOf<Int>()
            val first = async(start = CoroutineStart.UNDISPATCHED) {
                queue.submitRequest("first", "test", task = { started.complete(Unit); gate.await(); order.add(1) })
            }
            started.await()
            val second = async(start = CoroutineStart.UNDISPATCHED) {
                queue.submitRequest("second", "test", task = { order.add(2) })
            }
            val third = async(start = CoroutineStart.UNDISPATCHED) {
                queue.submitRequest("third", "test", task = { order.add(3) })
            }
            repeat(5) {
                assertEquals(2, queue.getQueueSize())
                assertTrue(queue.getQueueStats().isProcessing)
            }
            gate.complete(Unit)
            first.await(); second.await(); third.await()
            assertEquals(listOf(1, 2, 3), order)

            val nextStarted = CompletableDeferred<Unit>()
            val nextGate = CompletableDeferred<Unit>()
            val running = async(start = CoroutineStart.UNDISPATCHED) {
                queue.submitRequest("running", "test", task = { nextStarted.complete(Unit); nextGate.await() })
            }
            nextStarted.await()
            val removed = async(start = CoroutineStart.UNDISPATCHED) {
                runCatching { queue.submitRequest("removed", "test", task = { error("Must not run") }) }
            }
            queue.clearQueue()
            assertTrue(removed.await().exceptionOrNull() is IllegalStateException)
            assertEquals(0, queue.getQueueSize())
            nextGate.complete(Unit)
            running.await()
        }
    }
}
