// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android

import android.util.Log
import com.alibaba.mls.api.ApplicationProvider
import com.alibaba.mnnllm.android.ChatService.Companion.provide
import com.alibaba.mnnllm.android.chat.ChatDataItem
import com.alibaba.mnnllm.android.utils.FileUtils.getMmapDir
import com.alibaba.mnnllm.android.utils.ModelPreferences.useMmap
import com.alibaba.mnnllm.android.utils.ModelUtils.isR1Model
import com.taobao.meta.avatar.settings.MainSettings.getLlmPrompt
import java.io.File
import java.io.Serializable
import java.util.stream.Collectors
import kotlin.concurrent.Volatile

class ChatSession @JvmOverloads constructor(
    private val modelId: String,
    var sessionId: String,
    private val configPath: String,
    private val useTmpPath: Boolean,
    val savedHistory: List<ChatDataItem>?,
    private val isDiffusion: Boolean = false
) :
    Serializable {
    private var nativePtr: Long = 0

    @Volatile
    private var modelLoading = false

    @Volatile
    private var mGenerating = false

    @Volatile
    private var mReleaseRequeted = false

    private var keepHistory = false

    fun load() {
        Log.d(TAG, "MNN_DEBUG load begin")
        modelLoading = true
        var historyStringList: List<String>? = null
        if (this.savedHistory != null && !savedHistory.isEmpty()) {
            historyStringList =
                savedHistory.stream().map { obj: ChatDataItem -> obj.text }
                    .collect(Collectors.toList())
        }
        var rootCacheDir = ""
        if (useMmap(ApplicationProvider.get(), modelId)) {
            rootCacheDir = getMmapDir(modelId, configPath.contains("modelscope"))
            File(rootCacheDir).mkdirs()
        }
        rootCacheDir = ApplicationProvider.get().cacheDir.toString() + "/llm"
        File(rootCacheDir).mkdirs()
        val extraParams = HashMap<String, Any>()
        extraParams["system_prompt"] = getLlmPrompt(ApplicationProvider.get())
        nativePtr = initNative(
            "", modelId, configPath, useTmpPath,
            historyStringList, isDiffusion, isR1Model(modelId), extraParams
        )
        modelLoading = false
        if (mReleaseRequeted) {
            release()
        }
    }

    val debugInfo: String
        get() = getDebugInfoNative(nativePtr) + "\n"

    fun generateNewSession(): String {
        this.sessionId = System.currentTimeMillis().toString()
        return this.sessionId
    }

    fun generate(input: String, progressListener: GenerateProgressListener): HashMap<String, Any> {
        synchronized(this) {
            Log.d(TAG, "MNN_DEBUG submit$input")
            mGenerating = true
            val result = submitNative(nativePtr, input, keepHistory, progressListener)
            mGenerating = false
            if (mReleaseRequeted) {
                release()
            }
            return result
        }
    }

    fun generateDiffusion(
        input: String,
        output: String?,
        progressListener: GenerateProgressListener?
    ): HashMap<String, Any> {
        synchronized(this) {
            Log.d(TAG, "MNN_DEBUG submit$input")
            mGenerating = true
            val result = HashMap<String, Any>()
            return result
        }
    }

    fun reset() {
        synchronized(this) {
            resetNative(nativePtr)
        }
    }

    fun release() {
        synchronized(this) {
            Log.d(
                TAG,
                "MNN_DEBUG release nativePtr: $nativePtr mGenerating: $mGenerating"
            )
            if (!mGenerating && !modelLoading) {
                releaseInner()
            } else {
                mReleaseRequeted = true
                while (mGenerating || modelLoading) {
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
            releaseNative(nativePtr, isDiffusion)
            nativePtr = 0
            provide().removeSession(sessionId)
            (this as Object).notifyAll()
        }
    }

    external fun initNative(
        rootCacheDir: String?,
        modelId: String?,
        configPath: String?,
        useTmpPath: Boolean,
        history: List<String>?,
        isDiffusion: Boolean,
        isR1: Boolean,
        extraParams: HashMap<String, Any>?
    ): Long

    private external fun submitNative(
        instanceId: Long,
        input: String,
        keepHistory: Boolean,
        listener: GenerateProgressListener
    ): HashMap<String, Any>

    private external fun resetNative(instanceId: Long)

    private external fun getDebugInfoNative(instanceId: Long): String

    private external fun releaseNative(instanceId: Long, isDiffusion: Boolean)

    fun setKeepHistory(keepHistory: Boolean) {
        this.keepHistory = keepHistory
    }

    fun updatePrompt(llmPrompt: String) {
        val extraParams = HashMap<String, Any>()
        extraParams["system_prompt"] = llmPrompt
        updateConfigNative(nativePtr, extraParams)
    }

    external fun updateConfigNative(nativePtr: Long, extraParams: HashMap<String, Any>?)
    interface GenerateProgressListener {
        fun onProgress(progress: String?): Boolean
    }

    companion object {

        const val TAG: String = "ChatSession"
    }
}
