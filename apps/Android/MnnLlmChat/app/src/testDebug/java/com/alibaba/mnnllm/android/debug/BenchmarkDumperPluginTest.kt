package com.alibaba.mnnllm.android.debug

import com.alibaba.mnnllm.android.benchmark.BenchmarkProgress
import com.alibaba.mnnllm.android.benchmark.BenchmarkResult
import com.alibaba.mnnllm.android.benchmark.TestInstance
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.ByteArrayOutputStream
import java.io.PrintStream

class BenchmarkDumperPluginTest {

    private class FakeController : BenchmarkDebugController {
        var models = listOf(
            BenchmarkModelInfo("ModelScope/MNN/Qwen3___5-0___8B-MNN", true),
            BenchmarkModelInfo("ModelScope/MNN/Qwen2.5-1.5B-MNN", false)
        )
        var running = false
        var modelInitialized = false
        var currentModelId: String? = null
        var currentBackend = "cpu"
        var stopCalled = false
        var lastRunModelId: String? = null
        var lastRunBackend: String? = null
        var lastRunPrompt: Int = 0
        var lastRunGen: Int = 0
        var lastRunRepeat: Int = 0
        var simulateError = false

        override fun listModels(): List<BenchmarkModelInfo> = models

        override fun getStatus(): BenchmarkDebugStatus {
            return BenchmarkDebugStatus(
                isRunning = running,
                isModelInitialized = modelInitialized,
                currentModelId = currentModelId,
                currentBackend = currentBackend
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
            lastRunModelId = modelId
            lastRunBackend = backendType
            lastRunPrompt = prompt
            lastRunGen = gen
            lastRunRepeat = repeat

            if (simulateError) {
                errorCallback(0, "simulated error")
                return true
            }

            // Simulate progress
            progressCallback(BenchmarkProgress(50, "Running..."))

            // Simulate result
            val testInstance = TestInstance(
                modelConfigFile = "config.json",
                modelType = modelId,
                modelSize = 500_000_000L,
                threads = threads,
                useMmap = false,
                nPrompt = prompt,
                nGenerate = gen,
                backend = if (backendType == "opencl") 3 else 0,
                precision = 2,
                power = 0,
                memory = 2,
                dynamicOption = 0
            )
            testInstance.prefillUs.addAll(listOf(100_000L, 110_000L, 105_000L))
            testInstance.decodeUs.addAll(listOf(200_000L, 210_000L, 195_000L))

            completeCallback(BenchmarkResult(testInstance, true))
            return true
        }

        override fun stopBenchmark() {
            stopCalled = true
        }
    }

    @Test
    fun `plugin name should be benchmark`() {
        val plugin = BenchmarkDumperPlugin(FakeController())
        assertEquals("benchmark", plugin.name)
    }

    @Test
    fun `no args should print usage`() {
        val plugin = BenchmarkDumperPlugin(FakeController())
        val out = ByteArrayOutputStream()

        plugin.execute(emptyList(), PrintStream(out))

        val output = out.toString()
        assertTrue(output.contains("Usage: dumpapp benchmark"))
        assertTrue(output.contains("list"))
        assertTrue(output.contains("run"))
        assertTrue(output.contains("status"))
        assertTrue(output.contains("stop"))
    }

    @Test
    fun `list should show available models`() {
        val plugin = BenchmarkDumperPlugin(FakeController())
        val out = ByteArrayOutputStream()

        plugin.execute(listOf("list"), PrintStream(out))

        val output = out.toString()
        assertTrue(output.contains("Qwen3___5-0___8B-MNN"))
        assertTrue(output.contains("Qwen2.5-1.5B-MNN"))
        assertTrue(output.contains("[downloaded]"))
        assertTrue(output.contains("[not_downloaded]"))
    }

    @Test
    fun `list with no models should say no models`() {
        val controller = FakeController()
        controller.models = emptyList()
        val plugin = BenchmarkDumperPlugin(controller)
        val out = ByteArrayOutputStream()

        plugin.execute(listOf("list"), PrintStream(out))

        assertTrue(out.toString().contains("No models available"))
    }

    @Test
    fun `status should display benchmark state`() {
        val controller = FakeController()
        controller.running = true
        controller.modelInitialized = true
        controller.currentModelId = "Qwen3___5-0___8B-MNN"
        controller.currentBackend = "opencl"

        val plugin = BenchmarkDumperPlugin(controller)
        val out = ByteArrayOutputStream()

        plugin.execute(listOf("status"), PrintStream(out))

        val output = out.toString()
        assertTrue(output.contains("running: true"))
        assertTrue(output.contains("model_initialized: true"))
        assertTrue(output.contains("Qwen3___5-0___8B-MNN"))
        assertTrue(output.contains("opencl"))
    }

    @Test
    fun `stop should invoke controller`() {
        val controller = FakeController()
        val plugin = BenchmarkDumperPlugin(controller)
        val out = ByteArrayOutputStream()

        plugin.execute(listOf("stop"), PrintStream(out))

        assertTrue(controller.stopCalled)
        assertTrue(out.toString().contains("Stop requested"))
    }

    @Test
    fun `run without modelId should show usage`() {
        val plugin = BenchmarkDumperPlugin(FakeController())
        val out = ByteArrayOutputStream()

        plugin.execute(listOf("run"), PrintStream(out))

        assertTrue(out.toString().contains("Usage: dumpapp benchmark run"))
    }

    @Test
    fun `run should pass correct parameters`() {
        val controller = FakeController()
        val plugin = BenchmarkDumperPlugin(controller)
        val out = ByteArrayOutputStream()

        plugin.execute(
            listOf("run", "TestModel", "--backend", "opencl", "--prompt", "256", "--gen", "64", "--repeat", "5"),
            PrintStream(out)
        )

        assertEquals("TestModel", controller.lastRunModelId)
        assertEquals("opencl", controller.lastRunBackend)
        assertEquals(256, controller.lastRunPrompt)
        assertEquals(64, controller.lastRunGen)
        assertEquals(5, controller.lastRunRepeat)
    }

    @Test
    fun `run should show results on success`() {
        val controller = FakeController()
        val plugin = BenchmarkDumperPlugin(controller)
        val out = ByteArrayOutputStream()

        plugin.execute(listOf("run", "TestModel"), PrintStream(out))

        val output = out.toString()
        assertTrue(output.contains("=== LLM Benchmark ==="))
        assertTrue(output.contains("=== Result ==="))
        assertTrue(output.contains("Prefill:"))
        assertTrue(output.contains("Decode:"))
        assertTrue(output.contains("tok/s"))
    }

    @Test
    fun `run should show error on failure`() {
        val controller = FakeController()
        controller.simulateError = true
        val plugin = BenchmarkDumperPlugin(controller)
        val out = ByteArrayOutputStream()

        plugin.execute(listOf("run", "TestModel"), PrintStream(out))

        val output = out.toString()
        assertTrue(output.contains("[ERROR]"))
        assertTrue(output.contains("simulated error"))
    }

    @Test
    fun `run with default options should use defaults`() {
        val controller = FakeController()
        val plugin = BenchmarkDumperPlugin(controller)
        val out = ByteArrayOutputStream()

        plugin.execute(listOf("run", "TestModel"), PrintStream(out))

        assertEquals("TestModel", controller.lastRunModelId)
        assertEquals("cpu", controller.lastRunBackend)
        assertEquals(128, controller.lastRunPrompt)
        assertEquals(128, controller.lastRunGen)
        assertEquals(3, controller.lastRunRepeat)
    }
}
