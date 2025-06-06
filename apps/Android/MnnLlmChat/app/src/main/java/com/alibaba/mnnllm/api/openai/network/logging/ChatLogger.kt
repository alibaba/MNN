package com.alibaba.mnnllm.api.openai.network.logging

import OpenAIChatRequest
import io.ktor.server.application.ApplicationCall
import io.ktor.server.request.httpMethod
import io.ktor.server.request.path
import io.ktor.util.flattenEntries
import timber.log.Timber

/**
 * 聊天日志管理器
 * 负责集中管理聊天相关的日志记录
 */
class ChatLogger {

    companion object {
        private const val TAG_REQUEST = "RequestProcessing"
        private const val TAG_STREAM = "StreamResponse"
        private const val TAG_TRANSFORM = "MessageTransform"
    }

    /**
     * 记录请求开始
     */
    fun logRequestStart(traceId: String, call: ApplicationCall) {
        Timber.tag(TAG_REQUEST).d("[$traceId] 进入接口")
        Timber.tag(TAG_REQUEST).d("请求头: ${call.request.headers.flattenEntries().joinToString("; ")}")
        Timber.tag(TAG_REQUEST).i("收到请求: ${call.request.httpMethod.value} ${call.request.path()}")
    }

    /**
     * 记录请求体
     */
    fun logRequestBody(traceId: String, chatRequest: OpenAIChatRequest, rawBody: String? = null) {
        if (rawBody != null) {
            Timber.tag(TAG_REQUEST).d("原始请求体: $rawBody")
        }
        Timber.tag(TAG_REQUEST).d("[$traceId] 请求体: $chatRequest")
    }

    /**
     * 记录转换后的历史消息
     */
    fun logTransformedHistory(traceId: String, unifiedHistory: List<android.util.Pair<String, String>>) {
        Timber.tag(TAG_TRANSFORM).d("[$traceId] 转换后的统一历史消息: $unifiedHistory")
    }

    /**
     * 记录推理开始
     */
    fun logInferenceStart(traceId: String, historySize: Int) {
        Timber.tag(TAG_REQUEST).d("[$traceId] 使用API服务完整历史推理，消息数量: $historySize")
    }

    /**
     * 记录推理完成
     */
    fun logInferenceComplete(traceId: String) {
        Timber.tag(TAG_REQUEST).d("[$traceId] API服务推理完成")
    }

    /**
     * 记录流式响应
     */
    fun logStreamDelta(traceId: String, progress: String) {
        Timber.tag(TAG_STREAM).d("[$traceId] 发送delta: $progress")
    }

    /**
     * 记录流式响应结束
     */
    fun logStreamEnd(traceId: String) {
        Timber.tag(TAG_STREAM).d("[$traceId] 发送最终chunk")
        Timber.tag(TAG_STREAM).d("[$traceId] 发送 [DONE] 兼容标记")
    }

    /**
     * 记录错误
     */
    fun logError(traceId: String, error: Throwable, context: String = "") {
        val tag = when {
            context.contains("stream", ignoreCase = true) -> TAG_STREAM
            context.contains("transform", ignoreCase = true) -> TAG_TRANSFORM
            else -> TAG_REQUEST
        }
        Timber.tag(tag).e(error, "[$traceId] $context")
    }

    /**
     * 记录警告
     */
    fun logWarning(traceId: String, message: String, error: Throwable? = null) {
        if (error != null) {
            Timber.tag(TAG_STREAM).w(error, "[$traceId] $message")
        } else {
            Timber.tag(TAG_REQUEST).w("[$traceId] $message")
        }
    }

    /**
     * 记录LLM Session状态
     */
    fun logLlmSessionError(traceId: String) {
        Timber.tag(TAG_REQUEST).e("[$traceId] LlmSession为空")
    }
    
    /**
     * 记录信息日志
     */
    fun logInfo(traceId: String, message: String) {
        Timber.tag(TAG_REQUEST).i("[$traceId] $message")
    }
}