package com.alibaba.mnnllm.api.openai.network.logging

import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.asSharedFlow
import timber.log.Timber
import java.text.SimpleDateFormat
import java.util.*

/**
 * 日志收集器
 * 用于收集Timber日志并提供给UI显示
 */
class LogCollector private constructor() {
    
    companion object {
        @Volatile
        private var INSTANCE: LogCollector? = null
        
        fun getInstance(): LogCollector {
            return INSTANCE ?: synchronized(this) {
                INSTANCE ?: LogCollector().also { INSTANCE = it }
            }
        }
    }
    
    private val dateFormat = SimpleDateFormat("HH:mm:ss.SSS", Locale.getDefault())
    private val _logFlow = MutableSharedFlow<LogEntry>(replay = 100) // 保留最近100条日志
    val logFlow: SharedFlow<LogEntry> = _logFlow.asSharedFlow()
    
    /**
     * 日志条目数据类
     */
    data class LogEntry(
        val timestamp: String,
        val level: LogLevel,
        val tag: String,
        val message: String,
        val throwable: Throwable? = null,
        val fileName: String? = null,
        val lineNumber: Int? = null,
        val methodName: String? = null
    )
    
    /**
     * 日志级别枚举
     */
    enum class LogLevel {
        VERBOSE, DEBUG, INFO, WARN, ERROR
    }
    
    /**
     * 自定义Timber Tree，用于拦截日志并发送到Flow
     */
    inner class CollectorTree : Timber.Tree() {
        override fun log(priority: Int, tag: String?, message: String, t: Throwable?) {
            val level = when (priority) {
                android.util.Log.VERBOSE -> LogLevel.VERBOSE
                android.util.Log.DEBUG -> LogLevel.DEBUG
                android.util.Log.INFO -> LogLevel.INFO
                android.util.Log.WARN -> LogLevel.WARN
                android.util.Log.ERROR -> LogLevel.ERROR
                else -> LogLevel.DEBUG
            }
            
            // 获取调用栈信息
            val stackTrace = Thread.currentThread().stackTrace
            val callerInfo = findCallerInfo(stackTrace)
            
            val logEntry = LogEntry(
                timestamp = dateFormat.format(Date()),
                level = level,
                tag = tag ?: "Unknown",
                message = message,
                throwable = t,
                fileName = callerInfo?.fileName,
                lineNumber = callerInfo?.lineNumber,
                methodName = callerInfo?.methodName
            )
            
            // 只收集API相关的日志
            if (isApiRelatedLog(tag, message)) {
                _logFlow.tryEmit(logEntry)
            }
        }
        
        /**
         * 查找调用者信息
         */
        private fun findCallerInfo(stackTrace: Array<StackTraceElement>): CallerInfo? {
            // 跳过Timber和LogCollector相关的栈帧
            for (i in stackTrace.indices) {
                val element = stackTrace[i]
                val className = element.className
                
                // 跳过系统类、Timber类和LogCollector类
                if (!className.startsWith("timber.log") &&
                    !className.startsWith("com.alibaba.mnnllm.api.openai.network.logging") &&
                    !className.startsWith("java.lang") &&
                    !className.startsWith("android.") &&
                    className.contains("com.alibaba.mnnllm")) {
                    
                    return CallerInfo(
                        fileName = element.fileName ?: "Unknown",
                        lineNumber = element.lineNumber,
                        methodName = element.methodName,
                        className = className
                    )
                }
            }
            return null
        }
    }
    
    /**
     * 调用者信息数据类
     */
    data class CallerInfo(
        val fileName: String,
        val lineNumber: Int,
        val methodName: String,
        val className: String
    )
    
    /**
     * 判断是否为API相关日志
     */
    private fun isApiRelatedLog(tag: String?, message: String): Boolean {
        if (tag == null) return false
        
        return tag.contains("RequestProcessing", ignoreCase = true) ||
               tag.contains("StreamResponse", ignoreCase = true) ||
               tag.contains("MessageTransform", ignoreCase = true) ||
               message.contains("/v1/chat/completions", ignoreCase = true) ||
               message.contains("API", ignoreCase = true)
    }
    
    /**
     * 初始化日志收集器
     */
    fun initialize() {
        // 添加自定义Tree到Timber
        Timber.plant(CollectorTree())
    }
    
    /**
     * 手动添加日志条目
     */
    fun addLog(level: LogLevel, tag: String, message: String, throwable: Throwable? = null) {
        val logEntry = LogEntry(
            timestamp = dateFormat.format(Date()),
            level = level,
            tag = tag,
            message = message,
            throwable = throwable
        )
        _logFlow.tryEmit(logEntry)
    }
    
    /**
     * 格式化日志条目为字符串
     */
    fun formatLogEntry(entry: LogEntry): String {
        val levelChar = when (entry.level) {
            LogLevel.VERBOSE -> "V"
            LogLevel.DEBUG -> "D"
            LogLevel.INFO -> "I"
            LogLevel.WARN -> "W"
            LogLevel.ERROR -> "E"
        }
        
        val locationInfo = if (entry.fileName != null && entry.lineNumber != null) {
            " (${entry.fileName}:${entry.lineNumber})"
        } else ""
        
        return "[${entry.timestamp}] $levelChar/${entry.tag}$locationInfo: ${entry.message}" +
               if (entry.throwable != null) "\n${entry.throwable}" else ""
    }
    
    /**
     * 格式化日志条目为带有点击信息的字符串
     */
    fun formatLogEntryWithClickableInfo(entry: LogEntry): Pair<String, String?> {
        val formattedLog = formatLogEntry(entry)
        val clickableInfo = if (entry.fileName != null && entry.lineNumber != null) {
            "${entry.fileName}:${entry.lineNumber}"
        } else null
        
        return Pair(formattedLog, clickableInfo)
    }
}