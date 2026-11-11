// Created by ruoyi.sjd on 2025/5/7.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.llm;

import android.util.Log
import com.alibaba.mnnllm.android.llm.ChatService.Companion.provide
import com.alibaba.mnnllm.android.chat.model.ChatDataItem
import com.alibaba.mnnllm.android.modelsettings.ModelConfig
import com.alibaba.mnnllm.android.model.ModelTypeUtils
import com.alibaba.mnnllm.android.modelsettings.ModelConfig.Companion.getExtraConfigFile
import com.google.gson.Gson
import timber.log.Timber
import java.io.File
import java.util.concurrent.locks.ReentrantReadWriteLock
import kotlin.concurrent.Volatile
import kotlin.concurrent.read
import kotlin.concurrent.write
import android.util.Pair
import com.alibaba.mnnllm.android.utils.MmapUtils
import android.content.Context
import android.app.ActivityManager
import com.alibaba.mnnllm.android.modelsettings.Jinja
import com.alibaba.mnnllm.android.modelsettings.JinjaContext
import com.alibaba.mnnllm.android.modelsettings.ModelConfig.Companion.loadConfig
import com.alibaba.mnnllm.android.utils.FileSplitter
import com.alibaba.mnnllm.android.qnn.QnnModule
class LlmSession (
    private val modelId: String,
    override var sessionId: String,
    private val configPath: String,
    var savedHistory: List<ChatDataItem>?,
    var backendType: String? = null,
    private val useCustomConfig: Boolean = true
): ChatSession, ContextAwareSession {
    override var supportOmni: Boolean = false
    @Volatile
    private var nativePtr: Long = 0

    // Quick native calls (config setters, getters) run on the UI thread outside the session
    // monitor. They hold the read side so release() cannot free the native session under
    // them; release() takes the write side. Native never blocks inside these calls.
    private val nativeLock = ReentrantReadWriteLock()

    private inline fun <T> withNative(fallback: T, block: (Long) -> T): T = nativeLock.read {
        val ptr = nativePtr
        if (ptr == 0L) fallback else block(ptr)
    }

    @Volatile
    override var contextSummary: String? = null

    @Volatile
    override var contextAnchorId: Long = -1L

    @Volatile
    private var modelLoading = false

    @Volatile
    private var generating = false

    @Volatile
    private var releaseRequested = false

    private var keepHistory = false

    private var isQnn = false

    override fun getHistory(): List<ChatDataItem>?{
        return savedHistory
    }

    override fun setHistory(history: List<ChatDataItem>?) {
    }

    override fun load() {
        Log.d(TAG, "MNN_DEBUG load begin modelId: $modelId backend: $backendType")
        modelLoading = true
        isQnn = ModelTypeUtils.isQnnModel(modelId)

        checkAndMergeSplitFiles()
        var historyStringList: List<String>? = null
        var currentHistory = this.savedHistory
        if (!contextSummary.isNullOrEmpty() && !currentHistory.isNullOrEmpty()) {
            val anchorIndex = currentHistory.indexOfFirst { it.dbId == contextAnchorId }
            if (anchorIndex >= 0) {
                currentHistory = currentHistory.subList(anchorIndex, currentHistory.size)
            } else {
                // The summary would not line up with the history; native trimming applies instead.
                contextSummary = null
            }
        }
        if (!currentHistory.isNullOrEmpty()) {
            historyStringList = NativeHistory.alternate(currentHistory)
        }
        val config = if (useCustomConfig) {
            ModelConfig.loadMergedConfig(configPath, getExtraConfigFile(modelId))!!
        } else {
            ModelConfig.loadDefaultConfig(configPath)!!
        }
        var rootCacheDir: String? = ""
        if (config.useMmap == true) {
            rootCacheDir = MmapUtils.getMmapDir(modelId)
            File(rootCacheDir).mkdirs()
        }
        val configMap = HashMap<String, Any>().apply {
            put("is_r1", ModelTypeUtils.isR1Model(modelId))
            put("mmap_dir", rootCacheDir ?: "")
            put("keep_history", keepHistory)
            contextSummary?.takeIf { it.isNotEmpty() }?.let { put("context_summary", it) }
        }
        val llmConfig = if (useCustomConfig) {
            ModelConfig.loadMergedConfig(configPath, getExtraConfigFile(modelId))!!
        } else {
            ModelConfig.loadDefaultConfig(configPath)!!
        }
        // Override backend type from constructor only if not null
        if (backendType != null) {
            llmConfig.backendType = backendType
        }
        if (isQnn) {
            llmConfig.visualModel = "visual_qnn_${QnnModule.modelMiddleName()}.mnn"
        }
        Log.d(TAG, "MNN_DEBUG load initNative")
        nativePtr = initNative(
                configPath,
                historyStringList,
        if (llmConfig != null) {
            Gson().toJson(llmConfig)
        } else {
            "{}"
        },
        Gson().toJson(configMap)
        )
        Log.d(TAG, "MNN_DEBUG load initNative end")
        modelLoading = false
        if (nativePtr == 0L) {
            Log.e(TAG, "Model load failed - native initialization returned null pointer")
            throw IllegalStateException("Model load failed - the model module could not be loaded")
        }
        if (releaseRequested) {
            release()
        }
    }

    /**
     * Check if the model is successfully loaded and ready for inference
     */
    fun isModelLoaded(): Boolean {
        return nativePtr != 0L
    }
    
    /**
     * Check and merge split files for the current model
     */
    private fun checkAndMergeSplitFiles() {
        try {
            val configFile = File(configPath)
            val modelDir = configFile.parentFile
            
            if (modelDir != null && modelDir.exists()) {
                Log.d(TAG, "Checking for split files in model directory: ${modelDir.absolutePath}")
                
                if (FileSplitter.needsMerging(modelDir)) {
                    Log.d(TAG, "Found split files that need merging in ${modelDir.absolutePath}")
                    val success = FileSplitter.mergeAllSplitFiles(modelDir)
                    if (success) {
                        Log.d(TAG, "Successfully merged split files for model: $modelId")
                    } else {
                        Log.w(TAG, "Failed to merge some split files for model: $modelId")
                    }
                } else {
                    Log.d(TAG, "No split files found for model: $modelId")
                }
            } else {
                Log.w(TAG, "Model directory not found: ${modelDir?.absolutePath}")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error checking/merging split files for model: $modelId", e)
        }
    }

    fun getConfig(): ModelConfig? {
        return ModelConfig.loadMergedConfig(configPath, getExtraConfigFile(modelId))
    }

    private fun generateNewSessionId(): String {
        this.sessionId = System.currentTimeMillis().toString()
        return this.sessionId
    }

    override fun generate(prompt: String,
                          params: Map<String, Any>,
                          progressListener: GenerateProgressListener): HashMap<String, Any> {
        Log.d(TAG, "start generate prompt: $prompt")
        synchronized(this) {
            if (mockLatex) {
                Timber.d("MNN_DEBUG generate intercepted by mockLatex")
                return submitMockLatexHistory(progressListener)
            }
            Log.d(TAG, "MNN_DEBUG submit$prompt")
            generating = true
            val result = try {
                submitNative(nativePtr, prompt, keepHistory, progressListener)
            } finally {
                generating = false
            }
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
            nativeLock.write {
                releaseNative(nativePtr)
                nativePtr = 0
            }
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
        withNative(Unit) { updateEnableAudioOutputNative(it, enable) }
    }

    override val debugInfo
        get() = withNative("") { getDebugInfoNative(it) } + "\n"


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
        withNative(Unit) { updateMaxNewTokensNative(it, maxNewTokens) }
    }

    fun updateSystemPrompt(systemPrompt: String) {
        withNative(Unit) { updateSystemPromptNative(it, systemPrompt) }
    }

    override fun updateThinking(thinking: Boolean) {
        val loadedConfig = loadConfig(modelId)
        loadedConfig?.let {
            loadedConfig.jinja = Jinja(context = JinjaContext(enableThinking = thinking))
            ModelConfig.saveConfig(getExtraConfigFile(modelId), loadedConfig)
            updateConfig(Gson().toJson(loadedConfig))
        }
    }

    // Native queues config changes while a generation runs, so this never blocks the
    // calling (often UI) thread.
    fun updateConfig(configJson: String) {
        Log.d(TAG, "updateConfig: $configJson")
        withNative(Unit) { updateConfigNative(it, configJson) }
    }

    // The context methods share the session monitor with generate()/release(), so they
    // never run concurrently with a generation or after the native session is freed.
    override fun checkContext(prompt: String): ContextCheck = synchronized(this) {
        val ptr = nativePtr
        if (ptr == 0L) return ContextCheck.NONE
        val values = prepareContextNative(ptr, prompt)
        if (values == null || values.size < 4) return ContextCheck.NONE
        ContextCheck(values[0], values[1], values[2], ContextAction.fromCode(values[3]))
    }

    override fun compressContext(): ContextCompression = synchronized(this) {
        val ptr = nativePtr
        if (ptr == 0L) return ContextCompression(false, "", 0)
        generating = true
        try {
            val result = compressContextNative(ptr)
            ContextCompression(
                success = result["success"] == true,
                summary = result["summary"] as? String ?: "",
                keptUserTurns = (result["kept_user_turns"] as? Number)?.toInt() ?: 0,
            )
        } finally {
            generating = false
        }
    }

    override fun countContextTokens(): Int = synchronized(this) {
        val ptr = nativePtr
        if (ptr == 0L) 0 else countContextTokensNative(ptr)
    }

    override fun contextLimits(): ContextLimits = synchronized(this) {
        val ptr = nativePtr
        val values = if (ptr == 0L) null else contextLimitsNative(ptr)
        if (values == null || values.size < 2) ContextLimits(0, 0) else ContextLimits(values[0], values[1])
    }

    override fun cancelContextWork() {
        withNative(Unit) { cancelContextWorkNative(it) }
    }

    private external fun updateEnableAudioOutputNative(llmPtr: Long, enable: Boolean)


    private external fun updateMaxNewTokensNative(llmPtr: Long, maxNewTokens: Int)

    private external fun updateSystemPromptNative(llmPtr: Long, systemPrompt: String)

    private external fun updateAssistantPromptNative(llmPtr: Long, assistantPrompt: String)

    private external fun updateConfigNative(llmPtr: Long, configJson: String)

    private external fun prepareContextNative(llmPtr: Long, prompt: String): IntArray?

    private external fun compressContextNative(llmPtr: Long): HashMap<String, Any>

    private external fun countContextTokensNative(llmPtr: Long): Int

    private external fun contextLimitsNative(llmPtr: Long): IntArray?

    private external fun cancelContextWorkNative(llmPtr: Long)


    companion object {
        const val TAG: String = "LlmSession"
        var mockLatex: Boolean = false
        var mockLatexContent: String? = null

        init {
            System.loadLibrary("mnnllmapp")
        }
    }



    //New: public method supporting complete history messages
    fun submitFullHistory(
        history: List<Pair<String, String>>,
        progressListener: GenerateProgressListener
    ): HashMap<String, Any> {
        synchronized(this) {
            if (mockLatex) {
                Timber.d("MNN_DEBUG submitFullHistory intercepted by mockLatex")
                return submitMockLatexHistory(progressListener)
            }
            //Use Timber instead of Log
            Timber.d("MNN_DEBUG submitFullHistory with ${history.size} messages")
            //Type conversion: kotlin.Pair -> android.util.Pair
            val androidHistory = history.map { android.util.Pair(it.first, it.second) }
            //Call JNI method, remove unnecessary type conversion
            val result = submitFullHistoryNative(nativePtr, androidHistory, progressListener)
            generating = false
            return result
        }
    }

    private fun submitMockLatexHistory(progressListener: GenerateProgressListener): HashMap<String, Any> {
        val mockText = mockLatexContent ?: "Here is a math formula:\n\n\$E=mc^2$\n\nAnd a block formula:\n\n\$\$a^2 + b^2 = c^2\$\$\n\nEnd of mock."
        Thread {
            try {
                // Simulate streaming delay
                var index = 0
                val chunkSize = 3
                while (index < mockText.length) {
                    Thread.sleep(50)
                    val endIndex = Math.min(index + chunkSize, mockText.length)
                    val chunk = mockText.substring(index, endIndex)
                    if (progressListener.onProgress(chunk)) {
                        break
                    }
                    index = endIndex
                }
                progressListener.onProgress(null) // notify completion
            } catch (e: Exception) {
                Timber.e(e, "Mock generation failed")
            } finally {
                generating = false
            }
        }.start()
        val map = HashMap<String, Any>()
        map["success"] = true
        map["prompt_len"] = 10L
        map["decode_len"] = mockText.length.toLong()
        map["prefill_time"] = 100000L
        map["decode_time"] = 2000000L
        return map
    }
    private external fun submitFullHistoryNative(
        nativePtr: Long,
        history: List<android.util.Pair<String, String>>,
        progressListener: GenerateProgressListener
    ): HashMap<String, Any>

    fun modelId(): String {
        //Create temporary variable to avoid modifying original modelId
        return modelId

    }

    fun getSystemPrompt(): String? {
        return withNative(null) { getSystemPromptNative(it) }
    }

    private external fun getSystemPromptNative(llmPtr: Long): String?

    private external fun dumpConfigNative(llmPtr: Long): String

    fun dumpConfig(): String {
        return withNative("{}") { dumpConfigNative(it) }
    }

    // Helper function to get current memory usage in MB
    private fun getCurrentMemoryUsageMB(context: Context): Long {
        val runtime = Runtime.getRuntime()
        val usedMemoryBytes = runtime.totalMemory() - runtime.freeMemory()
        return usedMemoryBytes / (1024 * 1024) // Convert to MB
    }
    
    // Helper function to get total memory info
    private fun getMemoryInfo(context: Context): Pair<Long, Long> {
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memoryInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memoryInfo)
        
        val runtime = Runtime.getRuntime()
        val usedMemoryMB = (runtime.totalMemory() - runtime.freeMemory()) / (1024 * 1024)
        val availMemoryMB = memoryInfo.availMem / (1024 * 1024)
        
        return Pair(usedMemoryMB, availMemoryMB)
    }

    // Official benchmark functionality following llm_bench.cpp approach
    fun runBenchmark(
        context: Context,
        commandParams: com.alibaba.mnnllm.android.benchmark.CommandParameters,
        testInstance: com.alibaba.mnnllm.android.benchmark.TestInstance,
        callback: com.alibaba.mnnllm.android.benchmark.BenchmarkCallback
    ): com.alibaba.mnnllm.android.benchmark.BenchmarkResult {
        // Use coroutine instead of Thread for better lifecycle management
        return try {
            // Run the actual benchmark in C++ following llm_bench.cpp structure
            runBenchmarkNative(
                nativePtr, 
                commandParams.backend,
                commandParams.threads,
                commandParams.useMmap,
                commandParams.power,
                commandParams.precision,
                commandParams.memory,
                commandParams.dynamicOption,
                commandParams.nPrompt,
                commandParams.nGenerate,
                commandParams.nRepeat,
                commandParams.kvCache == "true",
                testInstance,
                callback
            )
        } catch (e: Exception) {
            com.alibaba.mnnllm.android.benchmark.BenchmarkResult(
                testInstance = testInstance,
                success = false,
                errorMessage = "benchmark failed: ${e.message}"
            )
        }
    }

    // C++ implementation following llm_bench.cpp approach
    private external fun runBenchmarkNative(
        nativePtr: Long,
        backend: Int,
        threads: Int,
        useMmap: Boolean,
        power: Int,
        precision: Int,
        memory: Int,
        dynamicOption: Int,
        nPrompt: Int,
        nGenerate: Int,
        nRepeat: Int,
        kvCache: Boolean,
        testInstance: com.alibaba.mnnllm.android.benchmark.TestInstance,
        callback: com.alibaba.mnnllm.android.benchmark.BenchmarkCallback
    ): com.alibaba.mnnllm.android.benchmark.BenchmarkResult

}