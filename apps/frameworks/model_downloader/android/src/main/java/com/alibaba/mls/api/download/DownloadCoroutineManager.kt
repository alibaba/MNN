// Created by ruoyi.sjd on 2025/1/27.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mls.api.download

import android.util.Log
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch

/**
 * Centralized coroutine management for download operations
 * Provides configurable scope and dispatcher to avoid threading issues
 * 
 * Usage:
 * ```
 * // Configure in Application.onCreate()
 * DownloadCoroutineManager.configureDispatcher(Dispatchers.IO) // or Dispatchers.Default
 * 
 * // In downloaders
 * DownloadCoroutineManager.launchDownload {
 *     // download logic
 * }
 * ```
 */
object DownloadCoroutineManager {
    private const val TAG = "DownloadCoroutineManager"
    
    /**
     * Configurable dispatcher for download operations
     * Default to Dispatchers.Default to avoid IO dispatcher blocking issues
     */
    var downloadDispatcher: CoroutineDispatcher = Dispatchers.Default
        private set
    
    /**
     * Coroutine scope for download operations
     */
    private val _downloadScope by lazy {
        CoroutineScope(downloadDispatcher + SupervisorJob())
    }
    
    /**
     * Configure the dispatcher used for download operations
     * @param dispatcher The coroutine dispatcher to use
     */
    fun configureDispatcher(dispatcher: CoroutineDispatcher) {
        Log.d(TAG, "Configuring download dispatcher to: $dispatcher")
        downloadDispatcher = dispatcher
    }
    
    /**
     * Initialize with IO dispatcher (call this when system is stable)
     */
    fun initializeWithIO() {
        configureDispatcher(Dispatchers.IO)
    }
    
    /**
     * Initialize with Default dispatcher (call this when IO dispatcher has issues)
     */
    fun initializeWithDefault() {
        configureDispatcher(Dispatchers.Default)
    }
    
    /**
     * Launch a download coroutine with proper error handling
     * @param block The suspend function to execute
     */
    fun launchDownload(block: suspend CoroutineScope.() -> Unit) {
        _downloadScope.launch(downloadDispatcher, block = block)
    }
    
    /**
     * Get a scope with download dispatcher for manual coroutine management
     */
    fun getDownloadScope(): CoroutineScope {
        return _downloadScope
    }
    
    /**
     * Reset to default configuration (for testing or recovery)
     */
    fun resetToDefault() {
        Log.d(TAG, "Resetting to default dispatcher (Dispatchers.Default)")
        downloadDispatcher = Dispatchers.Default
    }
    
    /**
     * Get current dispatcher info for debugging
     */
    fun getCurrentDispatcherInfo(): String {
        return "Current download dispatcher: $downloadDispatcher"
    }
} 