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
    fun initialize() {
        Timber.plant(FilteredDebugTree())
        
        val configInfo = when {
            LOG_ALL_TAGS -> "ALL TAGS (no filtering)"
            USE_BLACKLIST_MODE -> "BLACKLIST mode (blocked: ${blockedTimberTags.joinToString(", ")})"
            else -> "WHITELIST mode (allowed: ${allowedTimberTags.joinToString(", ")})"
        }
        
        Timber.tag("TimberConfig").d("Timber logging enabled with $configInfo")
    }
    
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
            
            if (shouldLog) {
                super.log(priority, tag, message, t)
            }
            // If shouldLog is false, do not print log
        }
    }
}
