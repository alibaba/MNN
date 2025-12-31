// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.utils

import timber.log.Timber

/**
 * Timber logging configuration manager
 * 
 * Configuration options:
 * 1. LOG_ALL_TAGS = true: Print all logs (ignore filtering)
 * 2. LOG_ALL_TAGS = false: Use filtering based on allowedTimberTags or blockedTimberTags
 * 
 * Filtering modes:
 * - WHITELIST mode: Only tags in allowedTimberTags will be printed
 * - BLACKLIST mode: All tags except those in blockedTimberTags will be printed
 */
object TimberConfig {
    
    // Set to true to print ALL logs (no filtering)
    private val LOG_ALL_TAGS = false
    
    // Set to true to use blacklist mode, false to use whitelist mode
    private val USE_BLACKLIST_MODE = true

    // Current log level (default to DEBUG)
    var currentLogLevel = android.util.Log.DEBUG
        private set

    fun setLogLevel(level: Int) {
        currentLogLevel = level
        Timber.tag("TimberConfig").d("Log level changed to: $level")
    }
    
    // Whitelist: Only these tags will be printed (when LOG_ALL_TAGS = false and USE_BLACKLIST_MODE = false)
    private val allowedTimberTags = arrayOf(
        "MnnLlmApplication",
        "ModelListManager", 
        "ModelListPresenter",
        // "ModelDownloadManager",
        "ApiServerConfig",
        "OpenAIService",
        "DebugActivity",
        "MainActivity",
        "ChatActivity"
        // You can add more allowed TAGs here
    )
    
    // Blacklist: These tags will NOT be printed (when LOG_ALL_TAGS = false and USE_BLACKLIST_MODE = true)
    private val blockedTimberTags = arrayOf(
        "ModelDownloadManager",
        "DownloadManager",
        "FileUtils"
        // You can add more blocked TAGs here
    )
    
    /**
     * Initialize Timber logging based on configuration
     */


    
    /**
     * Get current Timber configuration info
     */
    fun getConfigInfo(): String {
        return when {
            LOG_ALL_TAGS -> "ALL TAGS (no filtering)"
            USE_BLACKLIST_MODE -> "BLACKLIST mode (blocked: ${blockedTimberTags.joinToString(", ")})"
            else -> "WHITELIST mode (allowed: ${allowedTimberTags.joinToString(", ")})"
        }
    }

    /**
     * Check if specified TAG will be logged based on current configuration
     */
    fun isTagAllowed(tag: String): Boolean {
        return tag.contains("Model")
        // val currentTag = tag ?: "Unknown"
        
        // return when {
        //     LOG_ALL_TAGS -> true
        //     USE_BLACKLIST_MODE -> !blockedTimberTags.contains(currentTag)
        //     else -> allowedTimberTags.contains(currentTag)
        // }
    }
    
    /**
     * Get allowed tags list (for whitelist mode)
     */
    fun getAllowedTags(): Array<String> {
        return allowedTimberTags.copyOf()
    }
    
    /**
     * Get blocked tags list (for blacklist mode)
     */
    fun getBlockedTags(): Array<String> {
        return blockedTimberTags.copyOf()
    }
    
    /**
     * Check if logging all tags is enabled
     */
    fun isLogAllTags(): Boolean {
        return LOG_ALL_TAGS
    }
    
    /**
     * Check if blacklist mode is enabled
     */
    fun isBlacklistMode(): Boolean {
        return USE_BLACKLIST_MODE
    }

    // File logging configuration
    private const val MAX_FILE_SIZE = 1024 * 1024 // 1MB
    private const val MAX_FILE_COUNT = 10
    private const val BUFFER_SIZE = 20
    private const val LOG_DIR_NAME = "logs"
    private const val PREFS_NAME = "MnnLlmPrefs"
    private const val KEY_FILE_LOGGING_ENABLED = "file_logging_enabled"

    private var fileLoggingTree: FileLoggingTree? = null
    private var appContext: android.content.Context? = null

    /**
     * Initialize Timber logging based on configuration
     */
    fun initialize(context: android.content.Context) {
        appContext = context.applicationContext
        
        // Plant console tree
        Timber.plant(FilteredDebugTree())
        
        // Check preference for file logging
        val prefs = context.getSharedPreferences(PREFS_NAME, android.content.Context.MODE_PRIVATE)
        val isFileLoggingEnabled = prefs.getBoolean(KEY_FILE_LOGGING_ENABLED, false)
        
        if (isFileLoggingEnabled) {
            enableFileLogging(context)
        }
        
        val configInfo = when {
            LOG_ALL_TAGS -> "ALL TAGS (no filtering)"
            USE_BLACKLIST_MODE -> "BLACKLIST mode (blocked: ${blockedTimberTags.joinToString(", ")})"
            else -> "WHITELIST mode (allowed: ${allowedTimberTags.joinToString(", ")})"
        }
        
        Timber.tag("TimberConfig").d("Timber logging enabled with $configInfo. File logging: $isFileLoggingEnabled")
    }
    
    /**
     * Enable or disable file logging
     */
    fun setFileLoggingEnabled(enabled: Boolean) {
        val context = appContext ?: return
        val prefs = context.getSharedPreferences(PREFS_NAME, android.content.Context.MODE_PRIVATE)
        prefs.edit().putBoolean(KEY_FILE_LOGGING_ENABLED, enabled).apply()
        
        if (enabled) {
            enableFileLogging(context)
        } else {
            disableFileLogging()
        }
        Timber.tag("TimberConfig").d("File logging set to: $enabled")
    }
    
    fun isFileLoggingEnabled(): Boolean {
        val context = appContext ?: return false
        val prefs = context.getSharedPreferences(PREFS_NAME, android.content.Context.MODE_PRIVATE)
        return prefs.getBoolean(KEY_FILE_LOGGING_ENABLED, false)
    }
    
    private fun enableFileLogging(context: android.content.Context) {
        if (fileLoggingTree != null) return // Already enabled
        
        try {
            // Use internal files directory
            val logDir = java.io.File(context.filesDir, LOG_DIR_NAME)
            if (!logDir.exists()) {
                logDir.mkdirs()
            }
            val tree = FileLoggingTree(logDir)
            Timber.plant(tree)
            fileLoggingTree = tree
        } catch (e: Exception) {
            e.printStackTrace()
            Timber.tag("TimberConfig").e(e, "Failed to enable file logging")
        }
    }
    
    private fun disableFileLogging() {
        fileLoggingTree?.let {
            Timber.uproot(it)
            fileLoggingTree = null
        }
    }

    /**
     * Custom Timber Tree with flexible TAG filtering support
     */
    private class FilteredDebugTree : Timber.DebugTree() {
        override fun log(priority: Int, tag: String?, message: String, t: Throwable?) {
            val currentTag = tag ?: "Unknown"
            
            // If LOG_ALL_TAGS is true, print all logs
            if (LOG_ALL_TAGS) {
                super.log(priority, tag, message, t)
                return
            }
            
            // Apply filtering based on mode
            val shouldLog = when {
                USE_BLACKLIST_MODE -> {
                    // Blacklist mode: print all except blocked tags
                    !blockedTimberTags.contains(currentTag)
                }
                else -> {
                    // Whitelist mode: only print allowed tags
                    allowedTimberTags.contains(currentTag)
                }
            }
            
            if (shouldLog && priority >= currentLogLevel) {
                super.log(priority, tag, message, t)
            }
            // If shouldLog is false, do not print log
        }
    }

    /**
     * File Logging Tree that writes logs to disk with rotation support
     */
    private class FileLoggingTree(private val logDir: java.io.File) : Timber.Tree() {
        
        private val logQueue = java.util.concurrent.LinkedBlockingQueue<String>()
        private val writeHandlerThread = android.os.HandlerThread("AccessLogWriter")
        private var writeHandler: android.os.Handler
        private val simpleDateFormat = java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", java.util.Locale.getDefault())

        init {
            writeHandlerThread.start()
            writeHandler = android.os.Handler(writeHandlerThread.looper)
        }

        override fun log(priority: Int, tag: String?, message: String, t: Throwable?) {
            val currentTag = tag ?: "Unknown"
            val time = simpleDateFormat.format(java.util.Date())
            val priorityStr = when (priority) {
                android.util.Log.VERBOSE -> "V"
                android.util.Log.DEBUG -> "D"
                android.util.Log.INFO -> "I"
                android.util.Log.WARN -> "W"
                android.util.Log.ERROR -> "E"
                android.util.Log.ASSERT -> "A"
                else -> "?"
            }
            
            val logMessage = "$time $priorityStr/$currentTag: $message\n"
            logQueue.offer(logMessage)
            
            if (logQueue.size >= BUFFER_SIZE) {
                flush()
            }
        }
        
        private fun flush() {
            writeHandler.post {
                try {
                    val logsToWrite = java.util.ArrayList<String>()
                    logQueue.drainTo(logsToWrite)
                    
                    if (logsToWrite.isEmpty()) return@post
                    
                    val currentLogFile = getCurrentLogFile()
                    val writer = java.io.BufferedWriter(java.io.FileWriter(currentLogFile, true))
                    
                    for (log in logsToWrite) {
                        writer.write(log)
                    }
                    
                    writer.flush()
                    writer.close()
                    
                } catch (e: Exception) {
                    android.util.Log.e("FileLoggingTree", "Error writing logs to file", e)
                }
            }
        }
        
        private fun getCurrentLogFile(): java.io.File {
            // Find the current active log file (log.1 to log.10)
            // We want to write to the first one that is not full, or rotate if needed.
            // Requirement: "write to log.1 ... exceed 1M -> switch to log.2 ... until log.10 -> delete log.1 reuse"
            // This implies a circular buffer of FILES.
            // We need to know which is the "current" head.
            
            // Simple approach: Check log.1, if full -> log.2 ... 
            // But we need to remember where we were.
            // Let's check last modified or just keep an index in preferences?
            // Since we don't have prefs here easily without context (tho we could ask for it),
            // let's try to infer from file sizes check.
            
            // Actually, the requirement says: "write to log.1 ... exceed 1M -> switch to log.2 ... until log.10 -> delete log.1 reuse"
            // This is slightly ambiguous on restart. 
            // If I restart and log.1 is full, should I go to log.2? Yes.
            // If log.10 is full, should I go to log.1? Yes.
            
            var targetFile: java.io.File? = null
            
            // We iterate 1 to 10 to find the first one that is NOT full, OR the last modified one if all are "valid" but we want to continue appending?
            // The issue is distinguishing "full and move next" vs "appending".
            
            // Strategy:
            // maintain a marker or deduce.
            // If we just check 1..10:
            // 1. Find the last modified file within the set.
            // 2. If it exists and matches our "current" criteria.
            
            // Let's simply loop 1 to 10.
            for (i in 1..MAX_FILE_COUNT) {
                 val f = java.io.File(logDir, "log.$i")
                 if (f.exists()) {
                     if (f.length() < MAX_FILE_SIZE) {
                         // found a file that has space.
                         // BUT, is it the *next* logical file or just an old small file?
                         // If we are strictly filling 1, then 2, then 3...
                         // We should write to the 'latest' one.
                         
                         // If we have log.1 (full), log.2 (full), log.3 (empty). We should write to log.3.
                         // Use Last Modified to find the 'tip'.
                     }
                 }
            }
            
            // optimized approach:
            // Find the file with the most recent modification time.
            var newestFile: java.io.File? = null
            var newestIndex = 1
            var newestTime = 0L

            for (i in 1..MAX_FILE_COUNT) {
                val f = java.io.File(logDir, "log.$i")
                if (f.exists() && f.lastModified() > newestTime) {
                    newestTime = f.lastModified()
                    newestFile = f
                    newestIndex = i
                }
            }

            if (newestFile == null) {
                return java.io.File(logDir, "log.1")
            }

            // If newest file is full, move to next
            return if (newestFile.length() >= MAX_FILE_SIZE) {
                var nextIndex = newestIndex + 1
                if (nextIndex > MAX_FILE_COUNT) {
                    nextIndex = 1
                }
                val nextFile = java.io.File(logDir, "log.$nextIndex")
                // Delete if we are wrapping around (or if it exists) to start fresh because it's a new 'cycle' for this index
                if (nextFile.exists()) {
                    nextFile.delete() 
                }
                nextFile
            } else {
                newestFile
            }
        }
    }
}
