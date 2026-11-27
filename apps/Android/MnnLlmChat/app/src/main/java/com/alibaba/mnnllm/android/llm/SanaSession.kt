package com.alibaba.mnnllm.android.llm

import android.util.Log
import com.alibaba.mnnllm.android.chat.model.ChatDataItem
import com.alibaba.mnnllm.android.llm.ChatService.Companion.provide
import com.alibaba.mnnllm.android.modelsettings.ModelConfig
import com.google.gson.Gson
import kotlinx.coroutines.CancellationException

class SanaSession(
    private val modelId: String,
    override var sessionId: String,
    private val configPath: String,
    private var savedHistory: List<ChatDataItem>? = null
) : ChatSession {
    override var supportOmni: Boolean = false
    override val debugInfo: String = ""
    private var nativePtr: Long = 0
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
            val config = try {
                ModelConfig.loadConfig(modelId)
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                Log.w(TAG, "Failed to load ModelConfig for $modelId, using defaults: ${e.message}")
                null
            }
            val configMap = hashMapOf<String, Any>(
                "diffusion_memory_mode" to (config?.diffusionMemoryMode ?: "0"),
                "backend_type" to (config?.backendType ?: "opencl"),
                "image_width" to (config?.imageWidth ?: 512),
                "image_height" to (config?.imageHeight ?: 512),
                "grid_size" to (config?.gridSize ?: 1)
            )
            nativePtr = initNative(configPath, Gson().toJson(configMap))
            check(nativePtr != 0L) { "Native Sana session not initialized" }
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
            val result = generateNative(
                ptr,
                prompt,
                params["imageInput"] as? String ?: "",
                params["output"] as String,
                params["iterNum"] as Int,
                params["randomSeed"] as Int,
                params["useCfg"] as? Boolean ?: true,
                (params["cfgScale"] as? Number)?.toFloat() ?: 4.5f,
                progressListener
            ) ?: hashMapOf<String, Any>("error" to true, "message" to "Native generation returned null", "session_invalid" to true)
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

    private external fun initNative(resourcePath: String, configJson: String): Long
    private external fun releaseNative(instanceId: Long)
    private external fun generateNative(
        instanceId: Long,
        prompt: String,
        imagePath: String,
        outputPath: String,
        steps: Int,
        seed: Int,
        useCfg: Boolean,
        cfgScale: Float,
        progressListener: GenerateProgressListener
    ): HashMap<String, Any>?

    companion object {
        const val TAG = "SanaSession"
        init {
            System.loadLibrary("mnnllmapp")
        }
    }
}
