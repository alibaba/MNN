package com.alibaba.mnnllm.api.openai.network.routes

import com.alibaba.mnnllm.api.openai.network.queue.RequestQueueManager
import io.ktor.http.HttpStatusCode
import io.ktor.server.application.call
import io.ktor.server.response.respond
import io.ktor.server.routing.Route
import io.ktor.server.routing.get
import io.ktor.server.routing.delete
import kotlinx.serialization.Serializable

/**
 * 队列状态路由
 * 提供队列监控和管理功能
 */

/**
 * 队列状态响应数据类
 */
@Serializable
data class QueueStatusResponse(
    val status: String,
    val totalRequests: Long,
    val completedRequests: Long,
    val failedRequests: Long,
    val pendingRequests: Long,
    val isProcessing: Boolean,
    val successRate: Double,
    val message: String
)

/**
 * 注册队列相关的路由
 */
fun Route.queueRoutes() {
    val requestQueueManager = RequestQueueManager.getInstance()

    // 获取队列状态
    get("/v1/queue/status") {
        try {
            val stats = requestQueueManager.getQueueStats()
            
            val response = QueueStatusResponse(
                status = "ok",
                totalRequests = stats.totalRequests,
                completedRequests = stats.completedRequests,
                failedRequests = stats.failedRequests,
                pendingRequests = stats.pendingRequests,
                isProcessing = stats.isProcessing,
                successRate = stats.successRate,
                message = when {
                    stats.isProcessing -> "正在处理请求中"
                    stats.pendingRequests > 0 -> "队列中有 ${stats.pendingRequests} 个待处理请求"
                    else -> "队列空闲"
                }
            )
            
            call.respond(HttpStatusCode.OK, response)
        } catch (e: Exception) {
            call.respond(
                HttpStatusCode.InternalServerError,
                mapOf(
                    "status" to "error",
                    "message" to "获取队列状态失败: ${e.message}"
                )
            )
        }
    }
    
    // 清空队列（仅用于管理目的，需要认证）
    delete("/v1/queue/clear") {
        try {
            requestQueueManager.clearQueue()
            call.respond(
                HttpStatusCode.OK,
                mapOf(
                    "status" to "ok",
                    "message" to "队列已清空"
                )
            )
        } catch (e: Exception) {
            call.respond(
                HttpStatusCode.InternalServerError,
                mapOf(
                    "status" to "error",
                    "message" to "清空队列失败: ${e.message}"
                )
            )
        }
    }
}