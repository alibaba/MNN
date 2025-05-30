// Created by ruoyi.sjd on 2025/5/7.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.llm;

import android.util.Log
import com.alibaba.mls.api.ApplicationProvider
import com.alibaba.mnnllm.android.llm.ChatService.Companion.provide
import com.alibaba.mnnllm.android.chat.model.ChatDataItem
import com.alibaba.mnnllm.android.modelsettings.ModelConfig
import com.alibaba.mnnllm.android.utils.FileUtils
import com.alibaba.mnnllm.android.utils.ModelPreferences
import com.alibaba.mnnllm.android.model.ModelUtils
import com.alibaba.mnnllm.android.modelsettings.ModelConfig.Companion.getExtraConfigFile
import com.google.gson.Gson
import java.io.File
import java.util.stream.Collectors
import kotlin.concurrent.Volatile

class LlmSession (
    private val modelId: String,
    override var sessionId: String,
    private val configPath: String,
    val savedHistory: List<ChatDataItem>?,
): ChatSession{
    private var extraAssistantPrompt: String? = null
    override var supportOmni: Boolean = false
    private var nativePtr: Long = 0

    @Volatile
    private var modelLoading = false

    @Volatile
    private var generating = false

    @Volatile
    private var releaseRequested = false

    private var keepHistory = false

    override fun load() {
        Log.d(TAG, "MNN_DEBUG load begin")
        modelLoading = true
        var historyStringList: List<String>? = null
        if (!this.savedHistory.isNullOrEmpty()) {
            historyStringList =
                    savedHistory.stream()
                            .map { obj: ChatDataItem -> obj.text }
                    .filter { obj: String? -> obj != null }
                    .map { obj: String? -> obj!! }
                    .collect(Collectors.toList())
        }
        val config = ModelConfig.loadMergedConfig(configPath, getExtraConfigFile(modelId))!!
        var rootCacheDir: String? = ""
        if (config.useMmap == true) {
            rootCacheDir = FileUtils.getMmapDir(modelId, configPath.contains("modelscope"))
            File(rootCacheDir).mkdirs()
        }
        val backend = config.backendType
        val configMap = HashMap<String, Any>().apply {
            put("is_r1", ModelUtils.isR1Model(modelId))
            put("mmap_dir", rootCacheDir ?: "")
            put("keep_history", keepHistory)
        }
        val extraConfig = ModelConfig.loadMergedConfig(configPath, getExtraConfigFile(modelId))?.apply {
            this.assistantPromptTemplate = extraAssistantPrompt
            this.backendType = backend
        }
        Log.d(TAG, "MNN_DEBUG load initNative")
        nativePtr = initNative(
                configPath,
                historyStringList,
        if (extraConfig != null) {
            Gson().toJson(extraConfig)
        } else {
            "{}"
        },
        Gson().toJson(configMap)
        )
        Log.d(TAG, "MNN_DEBUG load initNative end")
        modelLoading = false
        if (releaseRequested) {
            release()
        }
    }

    private fun generateNewSessionId(): String {
        this.sessionId = System.currentTimeMillis().toString()
        return this.sessionId
    }

    override fun generate(prompt: String,
                          params: Map<String, Any>,
                          progressListener: GenerateProgressListener): HashMap<String, Any> {
        synchronized(this) {
            Log.d(TAG, "MNN_DEBUG submit$prompt")
            generating = true
            val result = submitNative(nativePtr, prompt, keepHistory, progressListener)
            generating = false
            if (releaseRequested) {
                release()
            }
            return result
        }
    }

    override fun reset(): String {
        synchronized(this) {
            resetNative(nativePtr)
        }
        return generateNewSessionId()
    }

    override fun release() {
        synchronized(this) {
            Log.d(
                    TAG,
                    "MNN_DEBUG release nativePtr: $nativePtr mGenerating: $generating"
            )
            if (!generating && !modelLoading) {
                releaseInner()
            } else {
                releaseRequested = true
                while (generating || modelLoading) {
                    try {
                        (this as Object).wait()
                    } catch (e: InterruptedException) {
                        Thread.currentThread().interrupt()
                        Log.e(TAG, "Thread interrupted while waiting for release", e)
                    }
                }
                releaseInner()
            }
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

    private external fun initNative(
            configPath: String?,
            history: List<String>?,
            mergedConfigStr: String?,
            configJsonStr: String?
    ): Long

    private external fun submitNative(
            instanceId: Long,
            input: String,
            keepHistory: Boolean,
            listener: GenerateProgressListener
    ): HashMap<String, Any>

    private external fun resetNative(instanceId: Long)

    private external fun getDebugInfoNative(instanceId: Long): String

    private external fun releaseNative(instanceId: Long)

    private external fun setWavformCallbackNative(
            instanceId: Long,
            listener: AudioDataListener?
    ): Boolean

    override fun setKeepHistory(keepHistory: Boolean) {
        this.keepHistory = keepHistory
    }

    override fun setEnableAudioOutput(enable: Boolean) {
        updateEnableAudioOutputNative(nativePtr, enable)
    }

    override val debugInfo
        get() = getDebugInfoNative(nativePtr) + "\n"


    fun setAudioDataListener(listener: AudioDataListener?) {
        synchronized(this) {
            if (nativePtr != 0L) {
                setWavformCallbackNative(nativePtr, listener)
            } else {
                Log.e(TAG, "nativePtr null")
            }
        }
    }

    fun updateMaxNewTokens(maxNewTokens: Int) {
        updateMaxNewTokensNative(nativePtr, maxNewTokens)
    }

    fun updateSystemPrompt(systemPrompt: String) {
        updateSystemPromptNative(nativePtr, systemPrompt)
    }

    fun updateAssistantPrompt(assistantPrompt: String) {
        extraAssistantPrompt = assistantPrompt
        updateAssistantPromptNative(nativePtr, assistantPrompt)
    }

    private external fun updateEnableAudioOutputNative(llmPtr: Long, enable: Boolean)


    private external fun updateMaxNewTokensNative(llmPtr: Long, maxNewTokens: Int)

    private external fun updateSystemPromptNative(llmPtr: Long, systemPrompt: String)

    private external fun updateAssistantPromptNative(llmPtr: Long, assistantPrompt: String)


    companion object {
        const val TAG: String = "LlmSession"

        init {
            System.loadLibrary("mnnllmapp")
        }
    }
}