// Created by ruoyi.sjd on 2025/5/7.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.llm

import android.util.Log
import com.alibaba.mls.api.ApplicationProvider
import com.alibaba.mnnllm.android.llm.ChatService.Companion.provide
import com.alibaba.mnnllm.android.llm.LlmSession.Companion.TAG
import com.alibaba.mnnllm.android.mainsettings.MainSettings.getDiffusionMemoryMode
import com.google.gson.Gson

class DiffusionSession(
    override var sessionId: String,
    private val configPath: String
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

    }

    override fun setKeepHistory(keepHistory: Boolean) {

    }

    override fun setEnableAudioOutput(enable: Boolean) {

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