// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android

import android.util.Log
import com.alibaba.mls.api.ApplicationProvider
import com.alibaba.mnnllm.android.ChatService.Companion.provide
import com.alibaba.mnnllm.android.chat.ChatDataItem
import com.alibaba.mnnllm.android.mainsettings.MainSettings.getDiffusionMemoryMode
import com.alibaba.mnnllm.android.modelsettings.ModelConfig
import com.alibaba.mnnllm.android.utils.FileUtils
import com.alibaba.mnnllm.android.utils.ModelPreferences
import com.alibaba.mnnllm.android.utils.ModelUtils
import com.google.gson.Gson
import org.json.JSONException
import org.json.JSONObject
import java.io.File
import java.util.stream.Collectors
import kotlin.concurrent.Volatile

class ChatSession @JvmOverloads constructor (
    private val modelId: String,
    var sessionId: String,
    private val configPath: String,
    val savedHistory: List<ChatDataItem>?,
    private val isDiffusion: Boolean = false
) {
    var supportOmni: Boolean = false

    private var nativePtr: Long = 0

    @Volatile
    private var modelLoading = false

    @Volatile
    private var generating = false

    @Volatile
    private var releaseRequeted = false

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
            File(rootCacheDir).mkdirs()
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
            configJson.put("mmap_dir", rootCacheDir?:"")
            configJson.put(
                "diffusion_memory_mode",
                getDiffusionMemoryMode(ApplicationProvider.get())
            )
        } catch (e: JSONException) {
            throw RuntimeException(e)
        }
        nativePtr = initNative(
            configPath,
            historyStringList, configJson.toString()
        )
        modelLoading = false
        if (releaseRequeted) {
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
            generating = true
            val result = submitNative(nativePtr, input, keepHistory, progressListener)
            generating = false
            if (releaseRequeted) {
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
            generating = true
            val result = submitDiffusionNative(
                nativePtr,
                input,
                output,
                iterNum,
                randomSeed,
                progressListener
            )
            generating = false
            if (releaseRequeted) {
                releaseInner()
            }
            return result
        }
    }

    fun reset() {
        synchronized(this) {
            resetNative(nativePtr, isDiffusion)
        }
    }

    fun release() {
        synchronized(this) {
            Log.d(
                TAG,
                "MNN_DEBUG release nativePtr: $nativePtr mGenerating: $generating"
            )
            if (!generating && !modelLoading) {
                releaseInner()
            } else {
                releaseRequeted = true
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

    fun loadConfig(): ModelConfig? {
        if (!isDiffusion && File(configPath).exists()) {
            val jsonString = File(configPath).readText()
            return Gson().fromJson(jsonString, ModelConfig::class.java)
        }
        return null
    }

    private fun releaseInner() {
        if (nativePtr != 0L) {
            releaseNative(nativePtr, isDiffusion)
            nativePtr = 0
            provide().removeSession(sessionId)
            (this as Object).notifyAll()
        }
    }

    private external fun initNative(
        configPath: String?,
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

    private external fun resetNative(instanceId: Long, isDiffusion: Boolean)

    private external fun getDebugInfoNative(instanceId: Long): String

    private external fun releaseNative(instanceId: Long, isDiffusion: Boolean)

    private external fun setWavformCallbackNative(
        instanceId: Long,
        listener: AudioDataListener?
    ): Boolean

    fun setKeepHistory(keepHistory: Boolean) {
        this.keepHistory = keepHistory
    }

    fun clearMmapCache() {
        FileUtils.clearMmapCache(modelId)
    }

    interface GenerateProgressListener {
        fun onProgress(progress: String?): Boolean
    }

    fun setAudioDataListener(listener: AudioDataListener?) {
        synchronized(this) {
            if (nativePtr != 0L) {
                setWavformCallbackNative(nativePtr, listener)
            } else {
                Log.e(TAG, "nativePtr null")
            }
        }
    }


    interface AudioDataListener {
        fun onAudioData(data: FloatArray, isEnd: Boolean): Boolean
    }

    companion object {
        const val TAG: String = "ChatSession"

        init {
            System.loadLibrary("mnnllmapp")
        }
    }
}
