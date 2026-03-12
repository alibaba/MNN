package com.alibaba.mnnllm.android.debug

import com.alibaba.mls.api.ApplicationProvider
import com.alibaba.mls.api.download.ModelDownloadManager
import com.alibaba.mnnllm.android.llm.DiffusionSession
import com.alibaba.mnnllm.android.llm.GenerateProgressListener
import com.alibaba.mnnllm.android.model.ModelTypeUtils
import com.alibaba.mnnllm.android.modelist.ModelListManager
import com.facebook.stetho.dumpapp.DumperContext
import com.facebook.stetho.dumpapp.DumperPlugin
import java.io.File
import java.io.PrintStream

internal data class DiffusionModelInfo(
    val modelId: String,
    val displayName: String,
    val isLocal: Boolean,
    val localPath: String?
)

internal data class DiffusionRunRequest(
    val modelRef: String,
    val prompt: String,
    val outputPath: String,
    val steps: Int,
    val seed: Int
)

internal data class DiffusionRunResult(
    val success: Boolean,
    val errorMessage: String? = null,
    val elapsedMs: Long = 0L,
    val outputPath: String? = null,
    val outputFileSize: Long? = null,
    val nativeResult: HashMap<String, Any>? = null
)

internal interface DiffusionDebugController {
    fun listModels(): List<DiffusionModelInfo>
    fun runGeneration(
        request: DiffusionRunRequest,
        progressCallback: (String) -> Unit
    ): DiffusionRunResult
}

internal object DefaultDiffusionDebugController : DiffusionDebugController {
    private data class ResolvedModel(val modelId: String, val modelPath: String)

    override fun listModels(): List<DiffusionModelInfo> {
        val state = ModelListManager.modelListState.value
        if (state !is ModelListManager.ModelListState.Success) {
            return emptyList()
        }

        return state.models.mapNotNull { wrapper ->
            val modelItem = wrapper.modelItem
            val modelId = modelItem.modelId ?: return@mapNotNull null
            val displayName = modelItem.modelName ?: modelId
            val isDiffusion = ModelTypeUtils.isDiffusionModel(displayName) || ModelTypeUtils.isDiffusionModel(modelId)
            val isSana = ModelTypeUtils.isSanaModel(displayName) || ModelTypeUtils.isSanaModel(modelId)
            if (!isDiffusion || isSana) {
                return@mapNotNull null
            }

            val localPath = modelItem.localPath ?: runCatching {
                ModelDownloadManager.getInstance(ApplicationProvider.get())
                    .getDownloadedFile(modelId)
                    ?.absolutePath
            }.getOrNull()

            DiffusionModelInfo(
                modelId = modelId,
                displayName = displayName,
                isLocal = wrapper.isLocal,
                localPath = localPath
            )
        }.sortedBy { it.modelId }
    }

    override fun runGeneration(
        request: DiffusionRunRequest,
        progressCallback: (String) -> Unit
    ): DiffusionRunResult {
        val resolved = resolveModel(request.modelRef)
            ?: return DiffusionRunResult(
                success = false,
                errorMessage = "Cannot resolve model path for: ${request.modelRef}"
            )

        val session = DiffusionSession(
            modelId = resolved.modelId,
            sessionId = System.currentTimeMillis().toString(),
            configPath = resolved.modelPath
        )
        val start = System.currentTimeMillis()

        return try {
            session.load()
            val nativeResult = session.generate(
                request.prompt,
                mapOf(
                    "output" to request.outputPath,
                    "iterNum" to request.steps,
                    "randomSeed" to request.seed
                ),
                object : GenerateProgressListener {
                    override fun onProgress(progress: String?): Boolean {
                        if (!progress.isNullOrEmpty()) {
                            progressCallback(progress)
                        }
                        return false
                    }
                }
            )
            val elapsedMs = System.currentTimeMillis() - start
            val hasError = nativeResult["error"] == true
            val outputFile = File(request.outputPath)
            DiffusionRunResult(
                success = !hasError,
                errorMessage = if (hasError) nativeResult["message"] as? String else null,
                elapsedMs = elapsedMs,
                outputPath = request.outputPath,
                outputFileSize = if (outputFile.exists()) outputFile.length() else null,
                nativeResult = nativeResult
            )
        } catch (e: Exception) {
            DiffusionRunResult(
                success = false,
                errorMessage = e.message ?: "Diffusion generation failed",
                elapsedMs = System.currentTimeMillis() - start
            )
        } finally {
            runCatching { session.release() }
        }
    }

    private fun resolveModel(modelRef: String): ResolvedModel? {
        if (modelRef.startsWith("local/")) {
            val localPath = modelRef.removePrefix("local/")
            if (File(localPath).exists()) {
                return ResolvedModel(modelRef, localPath)
            }
        }

        val directPath = File(modelRef)
        if (directPath.exists() && directPath.isDirectory) {
            return ResolvedModel("local/${directPath.absolutePath}", directPath.absolutePath)
        }

        val downloaded = runCatching {
            ModelDownloadManager.getInstance(ApplicationProvider.get()).getDownloadedFile(modelRef)
        }.getOrNull()
        if (downloaded != null && downloaded.exists()) {
            return ResolvedModel(modelRef, downloaded.absolutePath)
        }
        return null
    }
}

internal class DiffusionDumperPlugin(
    private val controller: DiffusionDebugController = DefaultDiffusionDebugController
) : DumperPlugin {
    override fun getName(): String = "diffusion"

    override fun dump(dumpContext: DumperContext) {
        execute(dumpContext.argsAsList, dumpContext.stdout)
    }

    internal fun execute(args: List<String>, writer: PrintStream) {
        if (args.isEmpty()) {
            printUsage(writer)
            return
        }

        when (args[0]) {
            "list" -> handleList(writer)
            "run" -> handleRun(writer, args.drop(1))
            else -> printUsage(writer)
        }
    }

    private fun handleList(writer: PrintStream) {
        val models = controller.listModels()
        if (models.isEmpty()) {
            writer.println("No diffusion models available")
            return
        }

        writer.println("Available diffusion models (${models.size}):")
        for (model in models) {
            writer.println("  ${model.modelId} [${if (model.isLocal) "downloaded" else "not_downloaded"}]")
            if (!model.localPath.isNullOrEmpty()) {
                writer.println("    path: ${model.localPath}")
            }
        }
    }

    private fun handleRun(writer: PrintStream, args: List<String>) {
        if (args.size < 2) {
            writer.println("Usage: dumpapp diffusion run <modelId|modelPath> <prompt> [options]")
            return
        }

        val modelRef = args[0]
        val prompt = args[1]
        var outputPath = "/sdcard/diffusion_output.jpg"
        var steps = 20
        var seed = 42

        var index = 2
        while (index < args.size) {
            when (args[index]) {
                "--output" -> {
                    outputPath = args.getOrElse(index + 1) { outputPath }
                    index += 2
                }
                "--steps" -> {
                    steps = args.getOrElse(index + 1) { steps.toString() }.toIntOrNull() ?: steps
                    index += 2
                }
                "--seed" -> {
                    seed = args.getOrElse(index + 1) { seed.toString() }.toIntOrNull() ?: seed
                    index += 2
                }
                else -> index++
            }
        }

        val request = DiffusionRunRequest(
            modelRef = modelRef,
            prompt = prompt,
            outputPath = outputPath,
            steps = steps,
            seed = seed
        )

        writer.println("=== Diffusion Generation ===")
        writer.println("Model: $modelRef")
        writer.println("Prompt: $prompt")
        writer.println("Output: $outputPath")
        writer.println("Steps: $steps")
        writer.println("Seed: $seed")
        writer.println()

        val result = controller.runGeneration(request) { progress ->
            writer.println("Progress: ${progress}%")
            writer.flush()
        }

        writer.println()
        writer.println("=== Result ===")
        writer.println("Success: ${result.success}")
        writer.println("Elapsed: ${result.elapsedMs}ms")
        if (!result.success) {
            writer.println("Error: ${result.errorMessage ?: "Unknown error"}")
            return
        }

        if (!result.outputPath.isNullOrEmpty()) {
            writer.println("Output saved to: ${result.outputPath}")
        }
        if (result.outputFileSize != null) {
            writer.println("Output file size: ${result.outputFileSize} bytes")
        }
        val totalTimeUs = result.nativeResult?.get("total_timeus")
        if (totalTimeUs != null) {
            writer.println("Native total_timeus: $totalTimeUs")
        }
    }

    private fun printUsage(writer: PrintStream) {
        writer.println("Usage: dumpapp diffusion <command>")
        writer.println("Commands:")
        writer.println("  list")
        writer.println("  run <modelId|modelPath> <prompt> [options]")
        writer.println()
        writer.println("Options for run:")
        writer.println("  --output <path>   Output image path (default: /sdcard/diffusion_output.jpg)")
        writer.println("  --steps <n>       Inference steps (default: 20)")
        writer.println("  --seed <n>        Random seed (default: 42)")
        writer.println()
        writer.println("Examples:")
        writer.println("  dumpapp diffusion list")
        writer.println("  dumpapp diffusion run ModelScope/MNN/stable-diffusion-v1-5 \"a cute cat\"")
        writer.println("  dumpapp diffusion run /data/local/tmp/sd15 \"a cyberpunk city\" --steps 30")
    }
}
