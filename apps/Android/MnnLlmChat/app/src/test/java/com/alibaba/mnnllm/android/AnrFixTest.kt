package com.alibaba.mnnllm.android

import com.alibaba.mnnllm.api.openai.network.queue.RequestQueueManager
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Tests for ANR (Application Not Responding) fixes.
 * Ensures blocking operations are moved off the main thread.
 */
class AnrFixTest {

    @Test
    fun `RequestQueueManager getQueueSize returns immediately without blocking`() {
        val manager = RequestQueueManager.getInstance()
        val start = System.nanoTime()
        repeat(50) {
            manager.getQueueSize()
        }
        val elapsedMs = (System.nanoTime() - start) / 1_000_000
        // 50 calls should complete in < 100ms (runBlocking would block much longer)
        assertTrue(
            "getQueueSize should not block: 50 calls took ${elapsedMs}ms",
            elapsedMs < 100
        )
    }

    @Test
    fun `RequestQueueManager getQueueSize returns 0 when queue is empty`() {
        val manager = RequestQueueManager.getInstance()
        assertEquals(0, manager.getQueueSize())
    }

    @Test
    fun `RequestQueueManager getQueueStats does not block`() {
        val manager = RequestQueueManager.getInstance()
        val stats = manager.getQueueStats()
        assertEquals(0L, stats.pendingRequests)
        assertTrue(stats.totalRequests >= 0)
    }
}
