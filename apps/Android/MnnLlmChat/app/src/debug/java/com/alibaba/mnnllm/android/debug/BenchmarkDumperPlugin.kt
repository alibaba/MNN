package com.alibaba.mnnllm.android.debug

import com.alibaba.mnnllm.android.benchmark.BenchmarkProgress
import com.alibaba.mnnllm.android.benchmark.BenchmarkResult
import com.facebook.stetho.dumpapp.DumperContext
import com.facebook.stetho.dumpapp.DumperPlugin
import android.util.Log
import java.io.PrintStream
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/**
 * Stetho DumperPlugin for running LLM benchmarks via adb command line.
 *
 * Usage:
 *   dumpapp benchmark list                         - List available models
 *   dumpapp benchmark run <modelId> [options]      - Run benchmark on a model
 *   dumpapp benchmark status                       - Show current benchmark status
 *   dumpapp benchmark stop                         - Stop running benchmark
 *
 * Options for 'run':
 *   --backend <cpu|opencl>    Backend type (default: cpu)
 *   --prompt <n>              Prompt token count (default: 128)
 *   --gen <n>                 Generate token count (default: 128)
 *   --repeat <n>              Repeat count (default: 3)
 *   --threads <n>             Thread count (default: 4)
 *
 * Examples:
 *   dumpapp benchmark list
 *   dumpapp benchmark run ModelScope/MNN/Qwen3___5-0___8B-MNN
 *   dumpapp benchmark run ModelScope/MNN/Qwen3___5-0___8B-MNN --backend opencl --repeat 5
 */
internal class BenchmarkDumperPlugin(
    private val controller: BenchmarkDebugController = DefaultBenchmarkDebugController
) : DumperPlugin {
    companion object {
        private const val TAG = "BenchmarkDumperPlugin"
    }

    override fun getName(): String = "benchmark"

    override fun dump(dumpContext: DumperContext) {
        execute(dumpContext.argsAsList, dumpContext.stdout)
    }

    internal fun execute(args: List<String>, writer: PrintStream) {
        if (args.isEmpty()) {
            doUsage(writer)
            return
        }

        when (args[0]) {
            "list" -> handleList(writer)
            "run" -> handleRun(writer, args.drop(1))
            "status" -> handleStatus(writer)
            "stop" -> handleStop(writer)
            else -> doUsage(writer)
        }
    }

    private fun handleList(writer: PrintStream) {
        val models = controller.listModels()
        if (models.isEmpty()) {
            writer.println("No models available")
            return
        }
        writer.println("Available models (${models.size}):")
        for (model in models) {
            writer.println("  ${model.modelId}  [${if (model.isLocal) "downloaded" else "not_downloaded"}]")
        }
    }

    private fun handleRun(writer: PrintStream, args: List<String>) {
        if (args.isEmpty()) {
            writer.println("Usage: dumpapp benchmark run <modelId> [options]")
            return
        }

        val modelId = args[0]

        // Parse options
        var backend = "cpu"
        var prompt = 128
        var gen = 128
        var repeat = 3
        var threads = 4

        var i = 1
        while (i < args.size) {
            when (args[i]) {
                "--backend" -> { backend = args.getOrElse(i + 1) { backend }; i += 2 }
                "--prompt" -> { prompt = args.getOrElse(i + 1) { "$prompt" }.toIntOrNull() ?: prompt; i += 2 }
                "--gen" -> { gen = args.getOrElse(i + 1) { "$gen" }.toIntOrNull() ?: gen; i += 2 }
                "--repeat" -> { repeat = args.getOrElse(i + 1) { "$repeat" }.toIntOrNull() ?: repeat; i += 2 }
                "--threads" -> { threads = args.getOrElse(i + 1) { "$threads" }.toIntOrNull() ?: threads; i += 2 }
                else -> i++
            }
        }

        val backendId = if (backend.equals("opencl", ignoreCase = true)) 3 else 0

        writer.println("=== LLM Benchmark ===")
        writer.println("Model:   $modelId")
        writer.println("Backend: $backend (id=$backendId)")
        writer.println("Prompt:  $prompt tokens")
        writer.println("Gen:     $gen tokens")
        writer.println("Repeat:  $repeat")
        writer.println("Threads: $threads")
        writer.println()

        val latch = CountDownLatch(1)
        var resultOutput = ""
        var writerClosed = false

        val runResult = controller.runBenchmark(
            modelId = modelId,
            backendType = backend,
            prompt = prompt,
            gen = gen,
            repeat = repeat,
            threads = threads,
            progressCallback = { progress ->
                if (!writerClosed) {
                    writerClosed = !safeWrite(writer, "[${progress.progress}%] ${progress.statusMessage}")
                }
            },
            completeCallback = { result ->
                resultOutput = formatResult(result)
                latch.countDown()
            },
            errorCallback = { errorCode, message ->
                resultOutput = "[ERROR] code=$errorCode: $message"
                latch.countDown()
            }
        )

        if (!runResult) {
            writer.println("[ERROR] Failed to start benchmark (model not found or already running)")
            return
        }

        // Wait up to 10 minutes
        val finished = latch.await(10, TimeUnit.MINUTES)
        if (!finished) {
            safeWrite(writer, "[TIMEOUT] Benchmark did not complete within 10 minutes")
            controller.stopBenchmark()
        } else {
            safeWrite(writer, "")
            safeWrite(writer, resultOutput)
        }
    }

    private fun safeWrite(writer: PrintStream, message: String): Boolean {
        return try {
            writer.println(message)
            writer.flush()
            true
        } catch (t: Throwable) {
            Log.w(TAG, "dumpapp output unavailable, continue benchmark without streaming output", t)
            false
        }
    }

    private fun handleStatus(writer: PrintStream) {
        val status = controller.getStatus()
        writer.println("Benchmark status:")
        writer.println("  running: ${status.isRunning}")
        writer.println("  model_initialized: ${status.isModelInitialized}")
        writer.println("  current_model: ${status.currentModelId ?: "none"}")
        writer.println("  current_backend: ${status.currentBackend}")
    }

    private fun handleStop(writer: PrintStream) {
        controller.stopBenchmark()
        writer.println("Stop requested")
    }

    private fun formatResult(result: BenchmarkResult): String {
        val sb = StringBuilder()
        sb.appendLine("=== Result ===")
        if (!result.success) {
            sb.appendLine("FAILED: ${result.errorMessage}")
            return sb.toString()
        }
        val t = result.testInstance
        sb.appendLine("Model:    ${t.modelType}")
        sb.appendLine("Backend:  ${if (t.backend == 3) "OpenCL" else "CPU"}")
        sb.appendLine("Threads:  ${t.threads}")
        sb.appendLine("Prompt:   pp${t.nPrompt}")
        sb.appendLine("Generate: tg${t.nGenerate}")

        if (t.prefillUs.isNotEmpty()) {
            val speeds = t.getTokensPerSecond(t.nPrompt, t.prefillUs)
            sb.appendLine("Prefill:  %.2f ± %.2f tok/s".format(t.getAvgUs(speeds), t.getStdevUs(speeds)))
        }
        if (t.decodeUs.isNotEmpty()) {
            val speeds = t.getTokensPerSecond(t.nGenerate, t.decodeUs)
            sb.appendLine("Decode:   %.2f ± %.2f tok/s".format(t.getAvgUs(speeds), t.getStdevUs(speeds)))
        }
        if (t.samplesUs.isNotEmpty()) {
            val totalTokens = t.nPrompt + t.nGenerate
            val speeds = t.getTokensPerSecond(totalTokens, t.samplesUs)
            sb.appendLine("Combined: %.2f ± %.2f tok/s".format(t.getAvgUs(speeds), t.getStdevUs(speeds)))
        }
        return sb.toString().trimEnd()
    }

    private fun doUsage(writer: PrintStream) {
        writer.println("Usage: dumpapp benchmark <command> [args]")
        writer.println()
        writer.println("Commands:")
        writer.println("  list                          List available models")
        writer.println("  run <modelId> [options]       Run benchmark on a model")
        writer.println("  status                        Show benchmark status")
        writer.println("  stop                          Stop running benchmark")
        writer.println()
        writer.println("Options for 'run':")
        writer.println("  --backend <cpu|opencl>        Backend (default: cpu)")
        writer.println("  --prompt <n>                  Prompt tokens (default: 128)")
        writer.println("  --gen <n>                     Generate tokens (default: 128)")
        writer.println("  --repeat <n>                  Repeat count (default: 3)")
        writer.println("  --threads <n>                 Thread count (default: 4)")
        writer.println()
        writer.println("Examples:")
        writer.println("  dumpapp benchmark list")
        writer.println("  dumpapp benchmark run ModelScope/MNN/Qwen3___5-0___8B-MNN")
        writer.println("  dumpapp benchmark run ModelScope/MNN/Qwen3___5-0___8B-MNN --backend opencl")
    }
}

internal data class BenchmarkDebugStatus(
    val isRunning: Boolean,
    val isModelInitialized: Boolean,
    val currentModelId: String?,
    val currentBackend: String
)

internal data class BenchmarkModelInfo(
    val modelId: String,
    val isLocal: Boolean
)

internal interface BenchmarkDebugController {
    fun listModels(): List<BenchmarkModelInfo>
    fun getStatus(): BenchmarkDebugStatus
    fun runBenchmark(
        modelId: String,
        backendType: String,
        prompt: Int,
        gen: Int,
        repeat: Int,
        threads: Int,
        progressCallback: (BenchmarkProgress) -> Unit,
        completeCallback: (BenchmarkResult) -> Unit,
        errorCallback: (Int, String) -> Unit
    ): Boolean
    fun stopBenchmark()
}
