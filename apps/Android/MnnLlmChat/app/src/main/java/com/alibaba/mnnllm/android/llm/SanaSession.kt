package com.alibaba.mnnllm.android.llm

import android.util.Log
import com.alibaba.mls.api.ApplicationProvider
import com.alibaba.mnnllm.android.chat.model.ChatDataItem
import com.alibaba.mnnllm.android.llm.ChatService.Companion.provide
import com.alibaba.mnnllm.android.modelsettings.ModelConfig
import com.google.gson.Gson
import java.util.HashMap

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
    @Volatile
    private var generating = false

    override fun load() {
        Log.d(TAG, "SanaSession load() called, configPath: $configPath")
        val config = try {
            ModelConfig.loadConfig(modelId)
        } catch (e: Exception) {
            Log.w(TAG, "Failed to load ModelConfig for $modelId, using defaults: ${e.message}")
            null
        }
        val configMap = HashMap<String, Any>().apply {
            put("diffusion_memory_mode", config?.diffusionMemoryMode ?: "0")
            put("backend_type", config?.backendType ?: "opencl")
            put("image_width", config?.imageWidth ?: 512)
            put("image_height", config?.imageHeight ?: 512)
            put("grid_size", config?.gridSize ?: 1)
        }
        nativePtr = initNative(
            configPath,
            Gson().toJson(configMap)
        )
        Log.d(TAG, "SanaSession load() nativePtr initialized: $nativePtr")
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
            Log.d(TAG, "Sana generate prompt: $prompt, nativePtr: $nativePtr")
            if (nativePtr == 0L) {
                Log.e(TAG, "nativePtr is 0, cannot generate")
                return HashMap<String, Any>().apply {
                    put("error", true)
                    put("message", "Native session not initialized")
                }
            }
            generating = true
            val output = params["output"] as String
            val imageInput = params["imageInput"] as? String ?: ""
            val steps = params["iterNum"] as Int
            val seed = params["randomSeed"] as Int
            val useCfg = params["useCfg"] as? Boolean ?: true
            val cfgScale = (params["cfgScale"] as? Number)?.toFloat() ?: 4.5f
            val result = generateNative(
                nativePtr,
                prompt,
                imageInput,
                output,
                steps,
                seed,
                useCfg,
                cfgScale,
                progressListener
            ) ?: HashMap<String, Any>().apply {
                put("error", true)
                put("message", "Native generation returned null")
            }

            // Check success flag from native
            if (result["success"] == false) {
                Log.e(TAG, "Native generation failed: ${result["message"]}")
            }

            generating = false
            if (releaseRequested) {
                releaseInner()
            }
            return result
        }
    }

    override fun reset(): String {
        return System.currentTimeMillis().toString().also { sessionId = it }
    }

    override fun release() {
        synchronized(this) {
            if (!generating) {
                releaseInner()
            } else {
                releaseRequested = true
            }
        }
    }
    
    private fun releaseInner() {
        if (nativePtr != 0L) {
            releaseNative(nativePtr)
            nativePtr = 0
            provide().removeSession(sessionId)
        }
    }

    override fun setKeepHistory(keepHistory: Boolean) {
        // Sana session does not support history
    }

    override fun setEnableAudioOutput(enable: Boolean) {
        // Not used
    }

    override fun getHistory(): List<ChatDataItem>? = savedHistory

    override fun setHistory(history: List<ChatDataItem>?) {
        savedHistory = history
    }

    override fun updateThinking(thinking: Boolean) {
    }

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