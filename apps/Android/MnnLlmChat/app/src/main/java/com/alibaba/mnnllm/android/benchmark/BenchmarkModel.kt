package com.alibaba.mnnllm.android.benchmark

import android.content.Context
import android.util.Log
import com.alibaba.mnnllm.android.modelist.ModelItemWrapper
import com.alibaba.mnnllm.android.modelist.ModelListManager
import java.text.SimpleDateFormat
import java.util.*
import kotlin.math.sqrt

/**
 * Model class for benchmark functionality
 * Handles data operations and business logic
 */
class BenchmarkModel {
    
    companion object {
        private const val TAG = "BenchmarkModel"
    }
    
    private val benchmarkService = BenchmarkService.getInstance()
    private val testResults = mutableListOf<TestInstance>()
    
    interface LoadModelsCallback {
        fun onSuccess(models: List<ModelItemWrapper>)
        fun onError(error: String)
    }

    interface BenchmarkModelCallback {
        fun onProgress(progress: BenchmarkProgress)
        fun onComplete(results: BenchmarkContract.BenchmarkResults)
        fun onBenchmarkModelError(errorCode: Int, message: String)
    }

    /**
     * Load available models from ModelListManager
     */
    suspend fun loadAvailableModels(context: Context): List<ModelItemWrapper> {
        return try {
            ModelListManager.loadAvailableModels(context)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load models", e)
            throw e
        }
    }
    
    /**
     * Start benchmark test with given parameters
     */
    fun startBenchmark(
        context: Context,
        modelWrapper: ModelItemWrapper,
        callback: BenchmarkModelCallback
    ) {
        val selectedModelId = modelWrapper.modelItem.modelId
        if (selectedModelId.isNullOrEmpty()) {
            callback.onBenchmarkModelError(BenchmarkErrorCode.MODEL_ERROR, "Invalid model selection")
            return
        }
        
        // Clear previous results
        testResults.clear()
        
        // Start memory monitoring
        MemoryMonitor.start()
        
        try {
            // Create runtime and test parameters following llm_bench.cpp defaults
            val runtimeParams = BenchmarkService.defaultRuntimeParams.copy(
                backends = listOf(0), // CPU only for mobile
                threads = listOf(4),
                precision = listOf(2), // Low precision
                memory = listOf(2) // Low memory
            )
            
            val testParams = BenchmarkService.defaultTestParams.copy(
                nPrompt = listOf(),
                nGenerate = listOf(),
                nPrompGen = listOf(Pair(128, 128)),
                nRepeat = listOf(3), // Reduced for mobile
                kvCache = "false" // llama-bench style test
            )
            benchmarkService.runBenchmark(
                context,
                selectedModelId,
                object : BenchmarkCallback {
                    override fun onProgress(progress: BenchmarkProgress) {
                        callback.onProgress(progress)
                    }
                    
                    override fun onComplete(result: BenchmarkResult) {
                        if (result.success) {
                            Log.d(TAG, "Kotlin onComplete received: " +
                                       "prefillUs size=${result.testInstance.prefillUs.size}, " +
                                       "decodeUs size=${result.testInstance.decodeUs.size}")
                            testResults.add(result.testInstance)
                            // Stop memory monitor and get max memory
                            MemoryMonitor.stop()
                            val maxMemoryKb = MemoryMonitor.getMaxMemoryPssKb()
                            MemoryMonitor.reset()
                            
                            // Create benchmark results
                            val benchmarkResults = createBenchmarkResults(
                                modelWrapper.displayName,
                                maxMemoryKb,
                                testResults
                            )
                            callback.onComplete(benchmarkResults)
                        }
                    }

                    override fun onBenchmarkError(errorCode:Int, message: String) {
                        MemoryMonitor.stop()
                        MemoryMonitor.reset()
                        callback.onBenchmarkModelError(errorCode, message)
                    }
                },
                runtimeParams,
                testParams
            )
        } catch (e: Exception) {
            MemoryMonitor.stop()
            MemoryMonitor.reset()
            callback.onBenchmarkModelError(BenchmarkErrorCode.BENCHMARK_FAILED_UNKOWN, e.message ?: "Unknown error")
        }
    }
    
    /**
     * Stop benchmark test
     */
    fun stopBenchmark() {
        benchmarkService.stopBenchmark()
        MemoryMonitor.stop()
        MemoryMonitor.reset()
    }
    
    /**
     * Check if benchmark is currently running
     */
    fun isBenchmarkRunning(): Boolean {
        return benchmarkService.isBenchmarkRunning()
    }
    
    /**
     * Check if model is initialized
     */
    fun isModelInitialized(): Boolean {
        return benchmarkService.isModelInitialized()
    }
    
    /**
     * Get current model info
     */
    fun getModelInfo(): String? {
        return benchmarkService.getModelInfo()
    }
    
    /**
     * Initialize model for benchmark
     */
    suspend fun initializeModel(modelId: String, configPath: String?): Boolean {
        return benchmarkService.initializeModel(modelId, configPath)
    }
    
    /**
     * Create benchmark results from test instances
     */
    private fun createBenchmarkResults(
        modelDisplayName: String,
        maxMemoryKb: Long,
        testResults: List<TestInstance>
    ): BenchmarkContract.BenchmarkResults {
        val timestamp = SimpleDateFormat("yyyy/M/dd HH:mm:ss", Locale.getDefault()).format(Date())
        
        return BenchmarkContract.BenchmarkResults(
            modelDisplayName = modelDisplayName,
            maxMemoryKb = maxMemoryKb,
            testResults = testResults,
            timestamp = timestamp
        )
    }
    
    /**
     * Calculate standard deviation for a list of doubles
     */
    fun calculateStdev(values: List<Double>): Double {
        if (values.size <= 1) return 0.0
        val mean = values.average()
        val sqSum = values.map { (it - mean) * (it - mean) }.sum()
        return sqrt(sqSum / (values.size - 1))
    }
    
    /**
     * Release resources
     */
    fun release() {
        benchmarkService.release()
    }
} 