// Created by ruoyi.sjd on 2025/5/7.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.llm

import android.util.Log
import com.alibaba.mls.api.ApplicationProvider
import com.alibaba.mnnllm.android.chat.model.ChatDataItem
import com.alibaba.mnnllm.android.llm.ChatService.Companion.provide
import com.alibaba.mnnllm.android.llm.LlmSession.Companion.TAG
import com.alibaba.mnnllm.android.mainsettings.MainSettings.getDiffusionMemoryMode
import com.google.gson.Gson

class DiffusionSession(
    override var sessionId: String,
    private val configPath: String,
    private var savedHistory: List<ChatDataItem>? = null
): ChatSession{
    override var supportOmni: Boolean = false
    private var nativePtr: Long = 0
    override val debugInfo: String = ""
    @Volatile
    private var releaseRequested = false
    @Volatile
    private var generating = false
    
    override fun load() {
        val configMap = HashMap<String, Any>().apply {
            put("diffusion_memory_mode", getDiffusionMemoryMode(ApplicationProvider.get()))
        }
        nativePtr = initNative(
            configPath,
            Gson().toJson(configMap)
        )
        if (releaseRequested) {
            release()
        }
    }

    override fun generate(
        prompt: String,
        params: Map<String, Any>,
        progressListener: GenerateProgressListener
    ): HashMap<String, Any> {
        synchronized(this) {
            Log.d(TAG, "MNN_DEBUG submit$prompt")
            generating = true
            val output = params["output"] as String
            val iterNum = params["iterNum"] as Int
            val randomSeed = params["randomSeed"] as Int
            val result = submitDiffusionNative(
                nativePtr,
                prompt,
                output,
                iterNum,
                randomSeed,
                progressListener
            )
            generating = false
            if (releaseRequested) {
                releaseInner()
            }
            return result
        }
    }

    private fun releaseInner() {
        if (nativePtr != 0L) {
            releaseNative(nativePtr)
            nativePtr = 0
            provide().removeSession(sessionId)
            (this as Object).notifyAll()
        }
    }


    private external fun submitDiffusionNative(
        instanceId: Long,
        input: String,
        outputPath: String,
        iterNum: Int,
        randomSeed: Int,
        progressListener: GenerateProgressListener
    ): HashMap<String, Any>

    private fun generateNewSessionId(): String {
        this.sessionId = System.currentTimeMillis().toString()
        return this.sessionId
    }

    override fun reset(): String {
        return generateNewSessionId()
    }

    override fun release() {
        synchronized(this) {
            Log.d(TAG, "MNN_DEBUG release nativePtr: $nativePtr generating: $generating")
            if (!generating) {
                releaseInner()
            } else {
                releaseRequested = true
            }
        }
    }

    override fun setKeepHistory(keepHistory: Boolean) {
        // Diffusion models don't use keepHistory for generation, but we still store it for UI
        // The history is managed by the UI layer, not the native layer
    }

    override fun setEnableAudioOutput(enable: Boolean) {
        // Diffusion models don't support audio output
    }

    override fun getHistory(): List<ChatDataItem>? {
        Log.d(TAG, "getHistory: returning ${savedHistory?.size ?: 0} items")
        return savedHistory
    }

    override fun setHistory(history: List<ChatDataItem>?) {
        Log.d(TAG, "setHistory: setting ${history?.size ?: 0} items")
        savedHistory = history
    }

    override fun updateThinking(thinking: Boolean) {
    }


    private external fun initNative(
        configPath: String,
        extraConfig:String
    ): Long

    private external fun resetNative(instanceId: Long)

    private external fun releaseNative(instanceId: Long)

    companion object {
        const val TAG: String = "DiffusionSession"

        init {
            System.loadLibrary("mnnllmapp")
        }
    }
}