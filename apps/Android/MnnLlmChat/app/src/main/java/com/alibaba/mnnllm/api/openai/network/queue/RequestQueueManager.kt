package com.alibaba.mnnllm.api.openai.network.queue

import kotlinx.coroutines.*
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import timber.log.Timber
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong

/**
 * 请求队列管理器
 * 确保LLM生成任务按顺序执行，同一时刻只能有一个任务运行
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
    
    // 请求队列
    private val requestQueue = Channel<QueuedRequest>(Channel.UNLIMITED)
    
    // 队列处理协程作用域
    private val queueScope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    
    // 当前是否正在处理请求
    private val isProcessing = AtomicInteger(0)
    
    // 队列统计
    private val totalRequests = AtomicLong(0)
    private val completedRequests = AtomicLong(0)
    private val failedRequests = AtomicLong(0)
    
    // 互斥锁用于保护队列状态
    private val queueMutex = Mutex()
    
    init {
        startQueueProcessor()
    }
    
    /**
     * 排队请求数据类
     */
    private data class QueuedRequest(
        val requestId: String,
        val traceId: String,
        val task: suspend () -> Unit,
        val onComplete: () -> Unit = {},
        val onError: (Exception) -> Unit = {},
        val priority: Int = 0 // 优先级，数字越小优先级越高
    )
    
    /**
     * 提交请求到队列
     * @param requestId 请求ID
     * @param traceId 追踪ID
     * @param task 要执行的任务
     * @param onComplete 完成回调
     * @param onError 错误回调
     * @param priority 优先级（可选，默认为0）
     */
    suspend fun submitRequest(
        requestId: String,
        traceId: String,
        task: suspend () -> Unit,
        onComplete: () -> Unit = {},
        onError: (Exception) -> Unit = {},
        priority: Int = 0
    ) {
        // 使用CompletableDeferred来等待任务完成
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
                "请求已加入队列: requestId=$requestId, traceId=$traceId, " +
                "队列长度=${getQueueSize()}, 优先级=$priority"
            )
            
            requestQueue.send(queuedRequest)
        }
        
        // 等待任务完成
        taskCompletion.await()
    }
    
    /**
     * 启动队列处理器
     */
    private fun startQueueProcessor() {
        queueScope.launch {
            Timber.tag("RequestQueue").i("请求队列处理器已启动")
            
            for (request in requestQueue) {
                try {
                    // 串行处理请求，无需compareAndSet检查
                    // 因为这个协程本身就是单线程的
                    processRequest(request)
                } catch (e: Exception) {
                    Timber.tag("RequestQueue").e(e, "队列处理器发生错误")
                    // 处理失败的请求
                    try {
                        request.onError(e)
                    } catch (callbackError: Exception) {
                        Timber.tag("RequestQueue").e(callbackError, "请求错误回调失败")
                    }
                }
            }
        }
    }
    
    /**
     * 处理单个请求
     */
    private suspend fun processRequest(request: QueuedRequest) {
        val startTime = System.currentTimeMillis()
        
        try {
            Timber.tag("RequestQueue").d(
                "开始处理请求: requestId=${request.requestId}, traceId=${request.traceId}"
            )
            
            // 执行实际任务
            request.task()
            
            // 任务完成
            completedRequests.incrementAndGet()
            request.onComplete()
            
            val duration = System.currentTimeMillis() - startTime
            Timber.tag("RequestQueue").d(
                "请求处理完成: requestId=${request.requestId}, " +
                "耗时=${duration}ms, 队列剩余=${getQueueSize()}"
            )
            
        } catch (e: Exception) {
            // 任务失败
            failedRequests.incrementAndGet()
            request.onError(e)
            
            val duration = System.currentTimeMillis() - startTime
            Timber.tag("RequestQueue").e(
                e, "请求处理失败: requestId=${request.requestId}, " +
                "耗时=${duration}ms, 错误: ${e.message}"
            )
        }
    }
    
    /**
     * 获取队列大小（近似值）
     */
    fun getQueueSize(): Int {
        return requestQueue.tryReceive().let {
            if (it.isSuccess) {
                // 如果能接收到，说明队列不为空，需要放回去
                runBlocking { requestQueue.send(it.getOrThrow()) }
                1 // 至少有一个
            } else {
                0 // 队列为空
            }
        }
    }
    
    /**
     * 获取队列统计信息
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
     * 队列统计信息
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
     * 清理队列（仅用于测试或紧急情况）
     */
    suspend fun clearQueue() {
        queueMutex.withLock {
            while (!requestQueue.isEmpty) {
                requestQueue.tryReceive()
            }
            Timber.tag("RequestQueue").w("队列已清空")
        }
    }
    
    /**
     * 关闭队列管理器
     */
    fun shutdown() {
        queueScope.cancel()
        Timber.tag("RequestQueue").i("请求队列管理器已关闭")
    }
}