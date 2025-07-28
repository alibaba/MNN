package com.alibaba.mnnllm.api.openai.network.queue

import kotlinx.coroutines.*
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import timber.log.Timber
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong

/**
 * Request queue manager
 * Ensures LLM generation tasks execute in order, only one task can run at the same time
 */
class RequestQueueManager private constructor() {
    
    companion object {
        @Volatile
        private var INSTANCE: RequestQueueManager? = null
        
        fun getInstance(): RequestQueueManager {
            return INSTANCE ?: synchronized(this) {
                INSTANCE ?: RequestQueueManager().also { INSTANCE = it }
            }
        }
    }
    
    // Request queue
    private val requestQueue = Channel<QueuedRequest>(Channel.UNLIMITED)
    
    // Queue processing coroutine scope
    private val queueScope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    
    // Whether currently processing requests
    private val isProcessing = AtomicInteger(0)
    
    // Queue statistics
    private val totalRequests = AtomicLong(0)
    private val completedRequests = AtomicLong(0)
    private val failedRequests = AtomicLong(0)
    
    // Mutex for protecting queue state
    private val queueMutex = Mutex()
    
    init {
        startQueueProcessor()
    }
    
    /**
     * Queued request data class
     */
    private data class QueuedRequest(
        val requestId: String,
        val traceId: String,
        val task: suspend () -> Unit,
        val onComplete: () -> Unit = {},
        val onError: (Exception) -> Unit = {},
        val priority: Int = 0 // Priority, smaller number means higher priority
    )
    
    /**
     * Submit request to queue
     * @param requestId Request ID
     * @param traceId Trace ID
     * @param task Task to execute
     * @param onComplete Completion callback
     * @param onError Error callback
     * @param priority Priority (optional, default is 0)
     */
    suspend fun submitRequest(
        requestId: String,
        traceId: String,
        task: suspend () -> Unit,
        onComplete: () -> Unit = {},
        onError: (Exception) -> Unit = {},
        priority: Int = 0
    ) {
        // Use CompletableDeferred to wait for task completion
        val taskCompletion = CompletableDeferred<Unit>()
        
        queueMutex.withLock {
            val queuedRequest = QueuedRequest(
                requestId = requestId,
                traceId = traceId,
                task = task,
                onComplete = {
                    onComplete()
                    taskCompletion.complete(Unit)
                },
                onError = { exception ->
                    onError(exception)
                    taskCompletion.completeExceptionally(exception)
                },
                priority = priority
            )
            
            totalRequests.incrementAndGet()
            
            Timber.tag("RequestQueue").d(
                "Request added to queue: requestId=$requestId, traceId=$traceId, " +
                "queue size=${getQueueSize()}, priority=$priority"
            )
            
            requestQueue.send(queuedRequest)
        }
        
        // Wait for task completion
        taskCompletion.await()
    }
    
    /**
     * Start queue processor
     */
    private fun startQueueProcessor() {
        queueScope.launch {
            Timber.tag("RequestQueue").i("Request queue processor started")
            
            for (request in requestQueue) {
                try {
                    // Process requests serially, no need for compareAndSet check
                    // Because this coroutine itself is single-threaded
                    processRequest(request)
                } catch (e: Exception) {
                    Timber.tag("RequestQueue").e(e, "Queue processor error occurred")
                    // Handle failed request
                    try {
                        request.onError(e)
                    } catch (callbackError: Exception) {
                        Timber.tag("RequestQueue").e(callbackError, "Request error callback failed")
                    }
                }
            }
        }
    }
    
    /**
     * Process single request
     */
    private suspend fun processRequest(request: QueuedRequest) {
        val startTime = System.currentTimeMillis()
        
        try {
            Timber.tag("RequestQueue").d(
                "Start processing request: requestId=${request.requestId}, traceId=${request.traceId}"
            )
            
            // Execute actual task
            request.task()
            
            // Task completed
            completedRequests.incrementAndGet()
            request.onComplete()
            
            val duration = System.currentTimeMillis() - startTime
            Timber.tag("RequestQueue").d(
                "Request processing completed: requestId=${request.requestId}, " +
                "duration=${duration}ms, remaining queue=${getQueueSize()}"
            )
            
        } catch (e: Exception) {
            // Task failed
            failedRequests.incrementAndGet()
            request.onError(e)
            
            val duration = System.currentTimeMillis() - startTime
            Timber.tag("RequestQueue").e(
                e, "Request processing failed: requestId=${request.requestId}, " +
                "duration=${duration}ms, error: ${e.message}"
            )
        }
    }
    
    /**
     * Get queue size (approximate value)
     */
    fun getQueueSize(): Int {
        return requestQueue.tryReceive().let {
            if (it.isSuccess) {
                // If can receive, queue is not empty, need to put it back
                runBlocking { requestQueue.send(it.getOrThrow()) }
                1 // At least one
            } else {
                0 // Queue is empty
            }
        }
    }
    
    /**
     * Get queue statistics
     */
    fun getQueueStats(): QueueStats {
        return QueueStats(
            totalRequests = totalRequests.get(),
            completedRequests = completedRequests.get(),
            failedRequests = failedRequests.get(),
            pendingRequests = getQueueSize().toLong(),
            isProcessing = isProcessing.get() == 1
        )
    }
    
    /**
     * Queue statistics
     */
    data class QueueStats(
        val totalRequests: Long,
        val completedRequests: Long,
        val failedRequests: Long,
        val pendingRequests: Long,
        val isProcessing: Boolean
    ) {
        val successRate: Double
            get() = if (totalRequests > 0) {
                completedRequests.toDouble() / totalRequests.toDouble() * 100
            } else 0.0
    }
    
    /**
     * Clear queue (for testing or emergency use only)
     */
    suspend fun clearQueue() {
        queueMutex.withLock {
            while (!requestQueue.isEmpty) {
                requestQueue.tryReceive()
            }
            Timber.tag("RequestQueue").w("Queue cleared")
        }
    }
    
    /**
     * Shutdown queue manager
     */
    fun shutdown() {
        queueScope.cancel()
        Timber.tag("RequestQueue").i("Request queue manager shutdown")
    }
}