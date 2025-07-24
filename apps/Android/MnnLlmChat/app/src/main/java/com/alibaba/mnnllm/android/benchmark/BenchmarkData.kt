package com.alibaba.mnnllm.android.benchmark

// Runtime parameters - following llm_bench.cpp RuntimeParameters structure
data class RuntimeParameters(
    val model: List<String> = listOf("./Qwen2.5-1.5B-Instruct"),
    val backends: List<Int> = listOf(0), // 0=CPU, 1=METAL, 3=OPENCL
    val threads: List<Int> = listOf(4),
    val useMmap: Boolean = false,
    val power: List<Int> = listOf(0), // 0=Normal, 1=High, 2=Low
    val precision: List<Int> = listOf(2), // 0=Normal, 1=High, 2=Low
    val memory: List<Int> = listOf(2), // 0=Normal, 1=High, 2=Low
    val dynamicOption: List<Int> = listOf(0)
)

// Test parameters - following llm_bench.cpp TestParameters structure
data class TestParameters(
    val nPrompt: List<Int> = listOf(512),
    val nGenerate: List<Int> = listOf(128),
    val nPrompGen: List<Pair<Int, Int>> = listOf(Pair(512, 128)),
    val nRepeat: List<Int> = listOf(5),
    val kvCache: String = "false", // "true" for llm_demo test, "false" for llama-bench test
    val loadTime: String = "false"
)

// Command parameters instance - following llm_bench.cpp CommandParameters structure
data class CommandParameters(
    val model: String,
    val backend: Int,
    val threads: Int,
    val useMmap: Boolean,
    val power: Int,
    val precision: Int,
    val memory: Int,
    val dynamicOption: Int,
    val nPrompt: Int,
    val nGenerate: Int,
    val nPrompGen: Pair<Int, Int>,
    val nRepeat: Int,
    val kvCache: String,
    val loadingTime: String
)

// Test instance - following llm_bench.cpp TestInstance structure
data class TestInstance(
    val modelConfigFile: String,
    val modelType: String,
    val modelSize: Long,
    val threads: Int,
    val useMmap: Boolean,
    val nPrompt: Int,
    val nGenerate: Int,
    val prefillUs: MutableList<Long> = mutableListOf(),
    val decodeUs: MutableList<Long> = mutableListOf(),
    val samplesUs: MutableList<Long> = mutableListOf(),
    val loadingS: MutableList<Double> = mutableListOf(),
    val backend: Int,
    val precision: Int,
    val power: Int,
    val memory: Int,
    val dynamicOption: Int
) {
    // Following llm_bench.cpp calculation methods
    fun getTokensPerSecond(nTokens: Int, costUs: List<Long>): List<Double> {
        return costUs.map { t -> 1e6 * nTokens / t }
    }
    
    fun getAvgUs(v: List<Double>): Double {
        return if (v.isEmpty()) 0.0 else v.average()
    }
    
    fun getStdevUs(v: List<Double>): Double {
        if (v.size <= 1) return 0.0
        val mean = v.average()
        val sqSum = v.map { (it - mean) * (it - mean) }.sum()
        return kotlin.math.sqrt(sqSum / (v.size - 1))
    }
}

// Progress update from C++
data class BenchmarkProgress(
    val progress: Int,      // 0-100
    val statusMessage: String, // Keep for backward compatibility
    // New structured fields for internationalization
    val progressType: ProgressType = ProgressType.UNKNOWN,
    val currentIteration: Int = 0,
    val totalIterations: Int = 0,
    val nPrompt: Int = 0,
    val nGenerate: Int = 0,
    val runTimeSeconds: Float = 0.0f,
    val prefillTimeSeconds: Float = 0.0f,
    val decodeTimeSeconds: Float = 0.0f,
    val prefillSpeed: Float = 0.0f,
    val decodeSpeed: Float = 0.0f
) {
    // Constructor for JNI layer - receives all parameters directly
    constructor(
        progress: Int,
        statusMessage: String,
        progressTypeInt: Int,
        currentIteration: Int,
        totalIterations: Int,
        nPrompt: Int,
        nGenerate: Int,
        runTimeSeconds: Float,
        prefillTimeSeconds: Float,
        decodeTimeSeconds: Float,
        prefillSpeed: Float,
        decodeSpeed: Float
    ) : this(
        progress,
        statusMessage,
        ProgressType.values().getOrNull(progressTypeInt) ?: ProgressType.UNKNOWN,
        currentIteration,
        totalIterations,
        nPrompt,
        nGenerate,
        runTimeSeconds,
        prefillTimeSeconds,
        decodeTimeSeconds,
        prefillSpeed,
        decodeSpeed
    )
}

// Progress type enumeration for structured progress reporting
enum class ProgressType {
    UNKNOWN,
    INITIALIZING,
    WARMING_UP,
    RUNNING_TEST,
    PROCESSING_RESULTS,
    COMPLETED,
    STOPPING
}

// Final benchmark result from C++ - updated to match official structure
data class BenchmarkResult(
    val testInstance: TestInstance,
    val success: Boolean,
    val errorMessage: String? = null
)

// Callback interface for C++ to call back to Kotlin
interface BenchmarkCallback {
    fun onProgress(progress: BenchmarkProgress)
    fun onComplete(result: BenchmarkResult)
    fun onBenchmarkError(errorCode:Int, message: String)
}

object BenchmarkErrorCode {
    const val BENCHMARK_FAILED_UNKOWN: Int = 30
    const val TEST_INSTANCE_FAILED: Int = 40
    const val MODEL_NOT_INITIALIZED: Int = 50
    const val BENCHMARK_RUNNING = 99
    const val BENCHMARK_STOPPED = 100
    const val NATIVE_ERROR = 0
    const val MODEL_ERROR = 2
}