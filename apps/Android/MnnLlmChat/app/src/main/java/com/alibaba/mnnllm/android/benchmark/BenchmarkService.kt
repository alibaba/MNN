package com.alibaba.mnnllm.android.benchmark

import android.content.Context
import android.util.Log
import com.alibaba.mnnllm.android.llm.ChatService
import com.alibaba.mnnllm.android.llm.LlmSession
import com.alibaba.mnnllm.android.modelsettings.ModelConfig
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class BenchmarkService {
    private var llmSession: LlmSession? = null
    private var isRunning = false
    @Volatile
    private var shouldStop = false

    companion object {
        private const val TAG = "BenchmarkService"
        
        @Volatile
        private var instance: BenchmarkService? = null
        
        fun getInstance(): BenchmarkService {
            return instance ?: synchronized(this) {
                instance ?: BenchmarkService().also { instance = it }
            }
        }
        
        // Default parameters following llm_bench.cpp defaults
        val defaultRuntimeParams = RuntimeParameters(
            model = listOf("./Qwen2.5-1.5B-Instruct"),
            backends = listOf(0), // CPU
            threads = listOf(4),
            useMmap = false,
            power = listOf(0),
            precision = listOf(2), // Low precision
            memory = listOf(2), // Low memory
            dynamicOption = listOf(0)
        )
        
        val defaultTestParams = TestParameters(
            nPrompt = listOf(512),
            nGenerate = listOf(128),
            nPrompGen = listOf(Pair(512, 128)),
            nRepeat = listOf(5),
            kvCache = "false", // llama-bench style test by default
            loadTime = "false"
        )
    }

    suspend fun initializeModel(modelId: String, customConfigPath: String? = null): Boolean = withContext(Dispatchers.IO) {
        try {
            Log.d(TAG, "Initializing benchmark model: $modelId")
            
            val configPath = customConfigPath ?: ModelConfig.getDefaultConfigFile(modelId)
            
            if (configPath.isNullOrEmpty()) {
                Log.e(TAG, "Config path not found for model: $modelId")
                return@withContext false
            }
            
            Log.d(TAG, "Using config path: $configPath")
            
            llmSession = ChatService.provide().createLlmSession(
                modelId,
                configPath,
                "benchmark_session_${System.currentTimeMillis()}",
                null,
                false
            ) as? LlmSession

            llmSession?.let { session ->
                session.load()
                Log.d(TAG, "Benchmark model loaded successfully")
                true
            } ?: false
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize benchmark model", e)
            false
        }
    }

    // Create command parameter instances following llm_bench.cpp approach
    private fun getCmdParamsInstances(
        runtimeParams: RuntimeParameters,
        testParams: TestParameters,
        modelConfigPath: String
    ): List<CommandParameters> {
        val instances = mutableListOf<CommandParameters>()
        
        // Following llm_bench.cpp: generate all combinations of parameters
        for (model in listOf(modelConfigPath)) { // Use actual model path
            for (backend in runtimeParams.backends) {
                for (threads in runtimeParams.threads) {
                    for (power in runtimeParams.power) {
                        for (precision in runtimeParams.precision) {
                            for (memory in runtimeParams.memory) {
                                for (dynamicOption in runtimeParams.dynamicOption) {
                                    // For each runtime configuration, create test configurations
                                    
                                    // nPrompt only tests (prefill performance)
                                    for (nPrompt in testParams.nPrompt) {
                                        instances.add(CommandParameters(
                                            model = model,
                                            backend = backend,
                                            threads = threads,
                                            useMmap = runtimeParams.useMmap,
                                            power = power,
                                            precision = precision,
                                            memory = memory,
                                            dynamicOption = dynamicOption,
                                            nPrompt = nPrompt,
                                            nGenerate = 0, // Only prefill
                                            nPrompGen = Pair(nPrompt, 0),
                                            nRepeat = testParams.nRepeat.first(),
                                            kvCache = testParams.kvCache,
                                            loadingTime = testParams.loadTime
                                        ))
                                    }
                                    
                                    // nGenerate only tests (decode performance)
                                    for (nGenerate in testParams.nGenerate) {
                                        instances.add(CommandParameters(
                                            model = model,
                                            backend = backend,
                                            threads = threads,
                                            useMmap = runtimeParams.useMmap,
                                            power = power,
                                            precision = precision,
                                            memory = memory,
                                            dynamicOption = dynamicOption,
                                            nPrompt = 0, // Only decode
                                            nGenerate = nGenerate,
                                            nPrompGen = Pair(0, nGenerate),
                                            nRepeat = testParams.nRepeat.first(),
                                            kvCache = testParams.kvCache,
                                            loadingTime = testParams.loadTime
                                        ))
                                    }
                                    
                                    // Combined prompt+generate tests
                                    for (nPrompGen in testParams.nPrompGen) {
                                        instances.add(CommandParameters(
                                            model = model,
                                            backend = backend,
                                            threads = threads,
                                            useMmap = runtimeParams.useMmap,
                                            power = power,
                                            precision = precision,
                                            memory = memory,
                                            dynamicOption = dynamicOption,
                                            nPrompt = nPrompGen.first,
                                            nGenerate = nPrompGen.second,
                                            nPrompGen = nPrompGen,
                                            nRepeat = testParams.nRepeat.first(),
                                            kvCache = testParams.kvCache,
                                            loadingTime = testParams.loadTime
                                        ))
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        return instances
    }

    private fun checkStop(callback: BenchmarkCallback):Boolean {
        if (shouldStop) {
            Log.d(TAG, "Benchmark stopped by user")
            isRunning = false
            CoroutineScope(Dispatchers.Main).launch {
                callback.onBenchmarkError(BenchmarkErrorCode.BENCHMARK_STOPPED,"Benchmark stopped by user")
            }
            false
        }
        return false
    }

    fun runBenchmark(
        context: Context,
        modelId: String,
        callback: BenchmarkCallback,
        runtimeParams: RuntimeParameters = defaultRuntimeParams,
        testParams: TestParameters = defaultTestParams
    ) {
        if (isRunning) {
            callback.onBenchmarkError(BenchmarkErrorCode.BENCHMARK_RUNNING,"Benchmark is already running")
            return
        }

        val session = llmSession
        if (session == null) {
            callback.onBenchmarkError(BenchmarkErrorCode.MODEL_NOT_INITIALIZED,"Model is not initialized")
            return
        }

        isRunning = true
        shouldStop = false
        Log.d(TAG, "Starting benchmark with official llm_bench.cpp approach")

        CoroutineScope(Dispatchers.IO).launch {
            try {
                val configPath = ModelConfig.getDefaultConfigFile(modelId) ?: ""
                val instances = getCmdParamsInstances(runtimeParams, testParams, configPath)
                
                Log.d(TAG, "Generated ${instances.size} test instances")
                
                var completedInstances = 0
                val totalInstances = instances.size
                
                for ((i, instance) in instances.withIndex()) {
                    // Check if benchmark should stop
                    Log.d(TAG, "running instance $i of $totalInstances")
                    if (checkStop(callback)) {
                        return@launch
                    }
                    try {
                        // Create TestInstance following llm_bench.cpp structure
                        val testInstance = TestInstance(
                            modelConfigFile = instance.model,
                            modelType = modelId,
                            modelSize = 0L, // Will be set in native code
                            threads = instance.threads,
                            useMmap = instance.useMmap,
                            nPrompt = instance.nPrompt,
                            nGenerate = instance.nGenerate,
                            backend = instance.backend,
                            precision = instance.precision,
                            power = instance.power,
                            memory = instance.memory,
                            dynamicOption = instance.dynamicOption
                        )
                        
                        // Update progress
                        val progress = (completedInstances * 100) / totalInstances
                        val statusMsg = "Running test ${completedInstances + 1}/$totalInstances: pp${instance.nPrompt}+tg${instance.nGenerate}"
                        Log.d(TAG, "Service Progress ($progress%): $statusMsg")
                        CoroutineScope(Dispatchers.Main).launch {
                            callback.onProgress(BenchmarkProgress(
                                progress = progress,
                                statusMessage = statusMsg
                            ))
                        }
                        
                        // Run the actual benchmark following llm_bench.cpp approach
                        val result = withContext(Dispatchers.Default) {
                            session.runBenchmark(
                                context,
                                instance,
                                testInstance,
                                object : BenchmarkCallback {
                                    override fun onProgress(progress: BenchmarkProgress) {
                                        if (shouldStop) {
                                            return
                                        }
                                        // Some native implementations may send negative or >100 values when finished
                                        val clampedProgress = progress.copy(progress = progress.progress.coerceIn(0, 100))
                                        // Filter extremely long nested logs for readability
                                        val shortMsg = if (clampedProgress.statusMessage.length > 256) {
                                            clampedProgress.statusMessage.take(256) + "…"
                                        } else clampedProgress.statusMessage

                                        Log.d(TAG, "Native Progress [${instance.nPrompt}p+${instance.nGenerate}g] (${clampedProgress.progress}%): $shortMsg")
                                        CoroutineScope(Dispatchers.Main).launch {
                                            callback.onProgress(clampedProgress)
                                        }
                                    }
                                    override fun onComplete(result: BenchmarkResult) {
                                    }

                                    override fun onBenchmarkError(errorCode:Int, message: String) {
                                        Log.e(TAG, "Native Error [${instance.nPrompt}p+${instance.nGenerate}g]: $message")
                                    }
                                }
                            )
                        }
                        if (checkStop(callback)) {
                            return@launch
                        }
                        completedInstances++
                        
                        // Log the raw result received from the native layer before passing it up.
                        if (result.success) {
                            Log.d(TAG, "Service received successful result from native: " +
                                       "prefillUs size=${result.testInstance.prefillUs.size}, " +
                                       "decodeUs size=${result.testInstance.decodeUs.size}")
                        } else {
                            Log.e(TAG, "Service received error result from native: ${result.errorMessage}")
                        }

                        // Return result for this instance
                        if (result.success) {
                            CoroutineScope(Dispatchers.Main).launch {
                                callback.onComplete(result)
                            }
                        }
                        
                    } catch (e: Exception) {
                        Log.e(TAG, "Error running test instance", e)
                        CoroutineScope(Dispatchers.Main).launch {
                            callback.onBenchmarkError(BenchmarkErrorCode.TEST_INSTANCE_FAILED,"Test instance failed: ${e.message}")
                        }
                    }
                }
                
                isRunning = false
                
            } catch (e: Exception) {
                isRunning = false
                Log.e(TAG, "Benchmark execution failed", e)
                CoroutineScope(Dispatchers.Main).launch {
                    callback.onBenchmarkError(BenchmarkErrorCode.BENCHMARK_FAILED_UNKOWN, "Benchmark failed: ${e.message}")
                }
            }
        }
    }

    fun release() {
        isRunning = false
        llmSession?.release()
        llmSession = null
        Log.d(TAG, "Benchmark service released")
    }

    fun isModelInitialized(): Boolean = llmSession != null

    fun getModelInfo(): String? = llmSession?.modelId()

    fun stopBenchmark() {
        Log.d(TAG, "benchmark stop request by the user")
        shouldStop = true
    }

    fun isBenchmarkRunning(): Boolean = isRunning
}