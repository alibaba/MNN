package com.alibaba.mnnllm.android.debug

import com.alibaba.mnnllm.android.benchmark.BenchmarkCallback
import com.alibaba.mnnllm.android.benchmark.BenchmarkProgress
import com.alibaba.mnnllm.android.benchmark.BenchmarkResult
import com.alibaba.mnnllm.android.benchmark.BenchmarkService
import com.alibaba.mnnllm.android.benchmark.RuntimeParameters
import com.alibaba.mnnllm.android.benchmark.TestParameters
import com.alibaba.mnnllm.android.model.ModelTypeUtils
import com.alibaba.mnnllm.android.modelist.ModelListManager
import kotlinx.coroutines.runBlocking

internal object DefaultBenchmarkDebugController : BenchmarkDebugController {

    private val service get() = BenchmarkService.getInstance()

    override fun listModels(): List<BenchmarkModelInfo> {
        val models = runBlocking { ModelListManager.getCurrentModels() } ?: return emptyList()
        return models
            .filterNot { ModelTypeUtils.isDiffusionModel(it.modelItem.modelName ?: "") }
            .mapNotNull { wrapper ->
                val id = wrapper.modelItem.modelId ?: return@mapNotNull null
                BenchmarkModelInfo(
                    modelId = id,
                    isLocal = wrapper.modelItem.isLocal
                )
            }
    }

    override fun getStatus(): BenchmarkDebugStatus {
        return BenchmarkDebugStatus(
            isRunning = service.isBenchmarkRunning(),
            isModelInitialized = service.isModelInitialized(),
            currentModelId = service.getModelInfo(),
            currentBackend = service.getCurrentBackendType()
        )
    }

    override fun runBenchmark(
        modelId: String,
        backendType: String,
        prompt: Int,
        gen: Int,
        repeat: Int,
        threads: Int,
        progressCallback: (BenchmarkProgress) -> Unit,
        completeCallback: (BenchmarkResult) -> Unit,
        errorCallback: (Int, String) -> Unit
    ): Boolean {
        if (service.isBenchmarkRunning()) {
            return false
        }

        // Initialize model if needed
        val initOk = runBlocking {
            service.initializeModel(modelId, backendType = backendType)
        }
        if (!initOk) {
            return false
        }

        val backendId = if (backendType.equals("opencl", ignoreCase = true)) 3 else 0

        val runtimeParams = RuntimeParameters(
            backends = listOf(backendId),
            threads = listOf(threads),
            precision = listOf(2),
            memory = listOf(2)
        )

        val testParams = TestParameters(
            nPrompt = emptyList(),
            nGenerate = emptyList(),
            nPrompGen = listOf(Pair(prompt, gen)),
            nRepeat = listOf(repeat),
            kvCache = "false"
        )

        service.runBenchmark(
            context = com.alibaba.mnnllm.android.MnnLlmApplication.getAppContext(),
            modelId = modelId,
            callback = object : BenchmarkCallback {
                override fun onProgress(progress: BenchmarkProgress) {
                    progressCallback(progress)
                }
                override fun onComplete(result: BenchmarkResult) {
                    completeCallback(result)
                }
                override fun onBenchmarkError(errorCode: Int, message: String) {
                    errorCallback(errorCode, message)
                }
            },
            runtimeParams = runtimeParams,
            testParams = testParams
        )
        return true
    }

    override fun stopBenchmark() {
        service.stopBenchmark()
    }
}
