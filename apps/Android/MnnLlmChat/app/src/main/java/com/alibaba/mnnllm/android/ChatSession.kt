// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android

import android.util.Log
import com.alibaba.mls.api.ApplicationProvider
import com.alibaba.mnnllm.android.ChatService.Companion.provide
import com.alibaba.mnnllm.android.chat.ChatDataItem
import com.alibaba.mnnllm.android.mainsettings.MainSettings.getDiffusionMemoryMode
import com.alibaba.mnnllm.android.utils.FileUtils
import com.alibaba.mnnllm.android.utils.ModelPreferences
import com.alibaba.mnnllm.android.utils.ModelUtils
import org.json.JSONException
import org.json.JSONObject
import java.io.File
import java.util.stream.Collectors
import kotlin.concurrent.Volatile

class ChatSession @JvmOverloads constructor (
    private val modelId: String,
    var sessionId: String,
    private val configPath: String,
    private val useTmpPath: Boolean,
    val savedHistory: List<ChatDataItem>?,
    private val isDiffusion: Boolean = false
) {
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
        if (!this.savedHistory.isNullOrEmpty()) {
            historyStringList =
                savedHistory.stream()
                    .map { obj: ChatDataItem -> obj.text }
                    .filter { obj: String? -> obj != null }
                    .map { obj: String? -> obj!! }
                    .collect(Collectors.toList())
        }
        var rootCacheDir: String? = ""
        if (ModelPreferences.useMmap(ApplicationProvider.get(), modelId)) {
            rootCacheDir = FileUtils.getMmapDir(modelId, configPath.contains("modelscope"))
            File(rootCacheDir!!).mkdirs()
        }
        val useOpencl = ModelPreferences.getBoolean(
            ApplicationProvider.get(),
            modelId, ModelPreferences.KEY_BACKEND, false
        )
        val backend = if (useOpencl) "opencl" else "cpu"
        val sampler = ModelPreferences.getString(
            ApplicationProvider.get(),
            modelId, ModelPreferences.KEY_SAMPLER, "greedy"
        )
        val configJson = JSONObject()
        try {
            configJson.put("backend", backend)
            configJson.put("sampler", sampler)
            configJson.put("is_diffusion", isDiffusion)
            configJson.put("is_r1", ModelUtils.isR1Model(modelId))
            configJson.put(
                "diffusion_memory_mode",
                getDiffusionMemoryMode(ApplicationProvider.get())
            )
        } catch (e: JSONException) {
            throw RuntimeException(e)
        }
        nativePtr = initNative(
            rootCacheDir, modelId, configPath,
            useTmpPath, historyStringList, configJson.toString()
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
        output: String,
        iterNum: Int,
        randomSeed: Int,
        progressListener: GenerateProgressListener
    ): HashMap<String, Any> {
        synchronized(this) {
            Log.d(TAG, "MNN_DEBUG submit$input")
            mGenerating = true
            val result = submitDiffusionNative(
                nativePtr,
                input,
                output,
                iterNum,
                randomSeed,
                progressListener
            )
            mGenerating = false
            if (mReleaseRequeted) {
                releaseInner()
            }
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
        configJsonStr: String?
    ): Long

    private external fun submitNative(
        instanceId: Long,
        input: String,
        keepHistory: Boolean,
        listener: GenerateProgressListener
    ): HashMap<String, Any>

    private external fun submitDiffusionNative(
        instanceId: Long,
        input: String,
        outputPath: String,
        iterNum: Int,
        randomSeed: Int,
        progressListener: GenerateProgressListener
    ): HashMap<String, Any>

    private external fun resetNative(instanceId: Long)

    private external fun getDebugInfoNative(instanceId: Long): String

    private external fun releaseNative(instanceId: Long, isDiffusion: Boolean)

    fun setKeepHistory(keepHistory: Boolean) {
        this.keepHistory = keepHistory
    }

    fun clearMmapCache() {
        FileUtils.clearMmapCache(modelId)
    }

    interface GenerateProgressListener {
        fun onProgress(progress: String?): Boolean
    }

    companion object {
        const val TAG: String = "ChatSession"

        init {
            System.loadLibrary("mnnllmapp")
            System.loadLibrary("llm")
            System.loadLibrary("MNN_CL")
        }
    }
}
