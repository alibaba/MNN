package com.alibaba.mnnllm.api.openai.network.logging

import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.asSharedFlow
import timber.log.Timber
import java.text.SimpleDateFormat
import java.util.*

/** * logcollector * forcollectTimberlogandprovide toUIdisplay*/
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
    private val _logFlow = MutableSharedFlow<LogEntry>(replay = 100) //preserve recent100 log entries
    val logFlow: SharedFlow<LogEntry> = _logFlow.asSharedFlow()
    
    /**
     * logentrydataclass*/
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
    
    /** * loglevelenum*/
    enum class LogLevel {
        VERBOSE, DEBUG, INFO, WARN, ERROR
    }
    
    /** * customTimber Tree，forinterceptlogandsendtoFlow*/
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
            
            //getcallstackinfo
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
            
            //onlycollectAPIrelatedlog
            if (isApiRelatedLog(tag, message)) {
                _logFlow.tryEmit(logEntry)
            }
        }
        
        /** * findcallerinfo*/
        private fun findCallerInfo(stackTrace: Array<StackTraceElement>): CallerInfo? {
            //skipTimberandLogCollectorrelatedstackframe
            for (i in stackTrace.indices) {
                val element = stackTrace[i]
                val className = element.className
                
                //skipsystemclass、TimberclassandLogCollectorclass
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
    
    /** * callerinfodataclass*/
    data class CallerInfo(
        val fileName: String,
        val lineNumber: Int,
        val methodName: String,
        val className: String
    )
    
    /** * determinewhether as API-relatedlog*/
    private fun isApiRelatedLog(tag: String?, message: String): Boolean {
        if (tag == null) return false
        
        return tag.contains("RequestProcessing", ignoreCase = true) ||
               tag.contains("StreamResponse", ignoreCase = true) ||
               tag.contains("MessageTransform", ignoreCase = true) ||
               message.contains("/v1/chat/completions", ignoreCase = true) ||
               message.contains("API", ignoreCase = true)
    }
    
    /** * initializelogcollector*/
    fun initialize() {
        //addcustomTreetoTimber
        Timber.plant(CollectorTree())
    }
    
    /** * manuallyaddlogentry*/
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
    
    /** * formatlogentryasstring*/
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
    
    /** * formatlog entryas withclickinfostring*/
    fun formatLogEntryWithClickableInfo(entry: LogEntry): Pair<String, String?> {
        val formattedLog = formatLogEntry(entry)
        val clickableInfo = if (entry.fileName != null && entry.lineNumber != null) {
            "${entry.fileName}:${entry.lineNumber}"
        } else null
        
        return Pair(formattedLog, clickableInfo)
    }
}