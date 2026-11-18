// Created by ruoyi.sjd on 2025/5/7.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.llm

import com.alibaba.mnnllm.android.chat.model.ChatDataItem
import com.alibaba.mnnllm.android.llm.ChatService.Companion.provide
import kotlinx.coroutines.CancellationException

class DiffusionSession(
    private val modelId: String,
    override var sessionId: String,
    private val configPath: String,
    private var savedHistory: List<ChatDataItem>? = null
) : ChatSession {
    override var supportOmni: Boolean = false
    private var nativePtr: Long = 0
    override val debugInfo: String = ""
    @Volatile
    private var releaseRequested = false
    private var releaseStarted = false
    private var generating = false
    private var modelLoading = false

    override fun load() {
        synchronized(this) {
            check(!releaseRequested) { "Session has been released" }
            check(!modelLoading && !generating) { "Session is busy" }
            if (nativePtr != 0L) return
            modelLoading = true
        }
        try {
            provide().beforeLoad(this)
            check(!releaseRequested) { "Session has been released" }
            nativePtr = initNative(
                configPath,
                DiffusionLoadConfigResolver.buildExtraConfigJson(modelId, configPath)
            )
            check(nativePtr != 0L) { "Native diffusion session not initialized" }
        } catch (e: Exception) {
            releaseRequested = true
            throw e
        } catch (e: OutOfMemoryError) {
            releaseRequested = true
            throw e
        } finally {
            synchronized(this) {
                modelLoading = false
                (this as Object).notifyAll()
            }
            if (releaseRequested) release()
        }
    }

    override fun generate(
        prompt: String,
        params: Map<String, Any>,
        progressListener: GenerateProgressListener
    ): HashMap<String, Any> {
        val ptr = synchronized(this) {
            check(!releaseRequested && nativePtr != 0L) { "Session is not available; reload the model" }
            check(!modelLoading && !generating) { "Session is busy" }
            generating = true
            nativePtr
        }
        try {
            val result = submitDiffusionNative(
                ptr,
                prompt,
                params["output"] as String,
                params["iterNum"] as Int,
                params["randomSeed"] as Int,
                progressListener
            ) ?: hashMapOf<String, Any>("error" to true, "message" to "Native diffusion returned null", "session_invalid" to true)
            if (result["session_invalid"] == true || result["error"] == true) releaseRequested = true
            return result
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            releaseRequested = true
            throw e
        } catch (e: OutOfMemoryError) {
            releaseRequested = true
            throw e
        } finally {
            synchronized(this) {
                generating = false
                (this as Object).notifyAll()
            }
            if (releaseRequested) release()
        }
    }

    override fun reset(): String = synchronized(this) {
        check(!releaseRequested && !generating && !modelLoading) { "Session is not available" }
        sessionId = System.currentTimeMillis().toString()
        provide().updateSessionId(this)
        sessionId
    }

    override fun release() {
        val ptr = synchronized(this) {
            releaseRequested = true
            if (generating || modelLoading || releaseStarted) return
            releaseStarted = true
            nativePtr.also { nativePtr = 0 }
        }
        var succeeded = false
        try {
            if (ptr != 0L) releaseNative(ptr)
            succeeded = true
        } finally {
            try {
                provide().removeSession(this)
            } finally {
                provide().onNativeReleased(this, succeeded)
                synchronized(this) { (this as Object).notifyAll() }
            }
        }
    }

    override fun setKeepHistory(keepHistory: Boolean) {}
    override fun setEnableAudioOutput(enable: Boolean) {}
    override fun getHistory(): List<ChatDataItem>? = savedHistory
    override fun setHistory(history: List<ChatDataItem>?) {
        savedHistory = history
    }
    override fun updateThinking(thinking: Boolean) {}

    private external fun submitDiffusionNative(
        instanceId: Long,
        input: String,
        outputPath: String,
        iterNum: Int,
        randomSeed: Int,
        progressListener: GenerateProgressListener
    ): HashMap<String, Any>?
    private external fun initNative(configPath: String, extraConfig: String): Long
    private external fun resetNative(instanceId: Long)
    private external fun releaseNative(instanceId: Long)

    companion object {
        const val TAG: String = "DiffusionSession"
        init {
            System.loadLibrary("mnnllmapp")
        }
    }
}
