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
import java.io.File
import java.util.concurrent.locks.ReentrantReadWriteLock
import kotlin.concurrent.Volatile
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
import kotlinx.coroutines.CancellationException
import org.json.JSONObject
class LlmSession (
    private val modelId: String,
    sessionId: String,
    private val configPath: String,
    var savedHistory: List<ChatDataItem>?,
    var backendType: String? = null,
    private val useCustomConfig: Boolean = true
): ChatSession, ContextAwareSession {
    @Volatile
    private var currentSessionId: String = sessionId
    override val sessionId: String get() = currentSessionId
    override var supportOmni: Boolean = false
    @Volatile
    private var nativePtr: Long = 0

    // Quick native calls (config setters, getters) run on the UI thread outside the session
    // monitor. They hold the read side so release() cannot free the native session under
    // them; release() takes the write side. Native never blocks inside these calls.
    private val nativeLock = ReentrantReadWriteLock()

    private inline fun <T> withNative(fallback: T, block: (Long) -> T): T {
        val lock = nativeLock.readLock()
        if (!lock.tryLock()) return fallback
        return try {
            val ptr = nativePtr
            if (ptr == 0L || releaseRequested || modelLoading) fallback else block(ptr)
        } finally {
            lock.unlock()
        }
    }

    private fun <T> withSessionOperation(block: (Long) -> T): T {
        val ptr = synchronized(this) {
            check(!releaseRequested && nativePtr != 0L) { "Session is not available; reload the model" }
            check(!modelLoading && !generating) { "Session is busy" }
            generating = true
            nativePtr
        }
        try {
            return block(ptr)
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

    private fun consumeResult(result: HashMap<String, Any>): HashMap<String, Any> {
        if (result["session_invalid"] == true) releaseRequested = true
        return result
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
    private var releaseStarted = false

    private var keepHistory = false
    @Volatile
    private var runtimeMaxNewTokens: Int? = null
    @Volatile
    private var runtimeAudioEnabled = false

    private var isQnn = false

    override fun getHistory(): List<ChatDataItem>?{
        return savedHistory
    }

    override fun setHistory(history: List<ChatDataItem>?) {
    }

    // Turns savedHistory into the flat native format. A restored summary only covers history
    // up to its anchor row, so anything before the anchor is dropped; a missing anchor means
    // the summary no longer matches and is discarded.
    private fun historyForNative(): List<String>? {
        var currentHistory = savedHistory
        if (!contextSummary.isNullOrEmpty() && !currentHistory.isNullOrEmpty()) {
            val anchorIndex = currentHistory.indexOfFirst { it.dbId == contextAnchorId }
            if (anchorIndex >= 0) {
                currentHistory = currentHistory.subList(anchorIndex, currentHistory.size)
            } else {
                contextSummary = null
            }
        }
        return currentHistory?.takeIf { it.isNotEmpty() }?.let { NativeHistory.alternate(it) }
    }

    /**
     * Rebinds this loaded session to another chat conversation without reloading the model:
     * engine KV is reset and the native history rebuilt from [history]. Worker thread only.
     */
    fun rebind(newSessionId: String, history: List<ChatDataItem>?) {
        withSessionOperation { ptr ->
            currentSessionId = newSessionId
            savedHistory = history
            replaceHistoryNative(ptr, historyForNative(), contextSummary ?: "")
        }
        provide().updateSessionId(this)
    }

    override fun load() {
        Log.d(TAG, "MNN_DEBUG load begin modelId: $modelId backend: $backendType")
        synchronized(this) {
            check(!releaseRequested) { "Session has been released" }
            check(!modelLoading && !generating) { "Session is busy" }
            if (nativePtr != 0L) return
            modelLoading = true
        }
        try {
            provide().beforeLoad(this) { releaseRequested }
            check(!releaseRequested) { "Session has been released" }
            isQnn = ModelTypeUtils.isQnnModel(modelId)
            checkAndMergeSplitFiles()
            val historyStringList = historyForNative()
            val llmConfig = if (useCustomConfig) {
                ModelConfig.loadMergedConfig(configPath, getExtraConfigFile(modelId))!!
            } else {
                ModelConfig.loadDefaultConfig(configPath)!!
            }
            var rootCacheDir: String? = ""
            if (llmConfig.useMmap == true) {
                rootCacheDir = MmapUtils.getMmapDir(modelId)
                File(rootCacheDir).mkdirs()
            }
            val configMap = HashMap<String, Any>().apply {
                put("is_r1", ModelTypeUtils.isR1Model(modelId))
                put("mmap_dir", rootCacheDir ?: "")
                put("keep_history", keepHistory)
                contextSummary?.takeIf { it.isNotEmpty() }?.let { put("context_summary", it) }
            }
            if (backendType != null) llmConfig.backendType = backendType
            if (isQnn) llmConfig.visualModel = "visual_qnn_${QnnModule.modelMiddleName()}.mnn"
            nativePtr = initNative(configPath, historyStringList, Gson().toJson(llmConfig), Gson().toJson(configMap))
            check(nativePtr != 0L) { "Model load failed - the model module could not be loaded" }
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

    /**
     * Check if the model is successfully loaded and ready for inference
     */
    fun isModelLoaded(): Boolean {
        return nativePtr != 0L && !modelLoading && !releaseRequested
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
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            Log.e(TAG, "Error checking/merging split files for model: $modelId", e)
        }
    }

    fun getConfig(): ModelConfig? {
        return ModelConfig.loadMergedConfig(configPath, getExtraConfigFile(modelId))
    }

    private fun generateNewSessionId(): String {
        currentSessionId = System.currentTimeMillis().toString()
        return currentSessionId
    }

    override fun generate(prompt: String,
                          params: Map<String, Any>,
                          progressListener: GenerateProgressListener): HashMap<String, Any> {
        Log.d(TAG, "start generate prompt: $prompt")
        return withSessionOperation { ptr ->
            if (mockLatex) {
                submitMockLatexHistory(progressListener)
            } else {
                consumeResult(submitNative(ptr, prompt, keepHistory, progressListener))
            }
        }
    }

    override fun reset(): String = withSessionOperation { ptr ->
        resetNative(ptr)
        generateNewSessionId().also { provide().updateSessionId(this) }
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
            if (ptr != 0L) nativeLock.write { releaseNative(ptr) }
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

    private external fun replaceHistoryNative(instanceId: Long, history: List<String>?, contextSummary: String)

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
        withNative(Unit) {
            updateEnableAudioOutputNative(it, enable)
            runtimeAudioEnabled = enable
        }
    }

    override val debugInfo
        get() = withNative("") { getDebugInfoNative(it) } + "\n"


    fun setAudioDataListener(listener: AudioDataListener?) {
        withNative(false) { setWavformCallbackNative(it, listener) }
    }

    fun updateMaxNewTokens(maxNewTokens: Int) {
        withNative(Unit) {
            updateMaxNewTokensNative(it, maxNewTokens)
            runtimeMaxNewTokens = maxNewTokens.coerceAtLeast(1)
        }
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

    data class AgentConfigSnapshot(
        val maxNewTokens: Int,
        val audioEnabled: Boolean,
        val jinja: String?,
        val expectsThinking: Boolean,
    )

    fun captureAgentConfig(): AgentConfigSnapshot {
        check(isModelLoaded()) { "Agent model is not available" }
        val config = JSONObject(dumpConfig())
        val jinja = config.optJSONObject("jinja")
        val thinking = jinja?.optJSONObject("context")?.opt("enable_thinking") as? Boolean
        return AgentConfigSnapshot(
            runtimeMaxNewTokens ?: config.optInt("max_new_tokens", 2048).coerceAtLeast(1),
            runtimeAudioEnabled, if (thinking != null) jinja.toString() else null, thinking == true
        )
    }

    fun configureAgentPass(snapshot: AgentConfigSnapshot, outputTokenLimit: Int, allowThinking: Boolean) {
        updateMaxNewTokens(outputTokenLimit)
        setEnableAudioOutput(false)
        snapshot.jinja?.let {
            val jinja = JSONObject(it)
            jinja.getJSONObject("context").put("enable_thinking", allowThinking)
            updateConfig(JSONObject().put("jinja", jinja).toString())
        }
    }

    fun restoreAgentConfig(snapshot: AgentConfigSnapshot) {
        updateMaxNewTokens(snapshot.maxNewTokens)
        snapshot.jinja?.let { updateConfig(JSONObject().put("jinja", JSONObject(it)).toString()) }
        setEnableAudioOutput(snapshot.audioEnabled)
    }

    override fun checkContext(prompt: String): ContextCheck = withSessionOperation { ptr ->
        val values = prepareContextNative(ptr, prompt)
        if (values == null || values.size < 4) ContextCheck.NONE else
            ContextCheck(values[0], values[1], values[2], ContextAction.fromCode(values[3]))
    }

    override fun compressContext(): ContextCompression = withSessionOperation { ptr ->
        val result = consumeResult(compressContextNative(ptr))
        ContextCompression(
            success = result["success"] == true,
            summary = result["summary"] as? String ?: "",
            keptUserTurns = (result["kept_user_turns"] as? Number)?.toInt() ?: 0,
        )
    }

    override fun countContextTokens(): Int = withSessionOperation { countContextTokensNative(it) }

    override fun contextLimits(): ContextLimits = withSessionOperation { ptr ->
        val values = contextLimitsNative(ptr)
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
        return withSessionOperation { ptr ->
            if (mockLatex) {
                submitMockLatexHistory(progressListener)
            } else {
                val androidHistory = history.map { android.util.Pair(it.first, it.second) }
                consumeResult(submitFullHistoryNative(ptr, androidHistory, progressListener))
            }
        }
    }

    private fun submitMockLatexHistory(progressListener: GenerateProgressListener): HashMap<String, Any> {
        val mockText = mockLatexContent ?: "Here is a math formula:\n\n\$E=mc^2$\n\nAnd a block formula:\n\n\$\$a^2 + b^2 = c^2\$\$\n\nEnd of mock."
        var index = 0
        val chunkSize = 3
        while (index < mockText.length) {
            Thread.sleep(50)
            val endIndex = Math.min(index + chunkSize, mockText.length)
            val chunk = mockText.substring(index, endIndex)
            if (progressListener.onProgress(chunk)) break
            index = endIndex
        }
        progressListener.onProgress(null)
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
        return try {
            withSessionOperation { ptr ->
                runBenchmarkNative(
                    ptr,
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
            }
        } catch (e: CancellationException) {
            throw e
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