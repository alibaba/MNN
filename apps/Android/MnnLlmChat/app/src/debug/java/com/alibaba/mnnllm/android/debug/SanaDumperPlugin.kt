package com.alibaba.mnnllm.android.debug

import android.os.Environment
import com.alibaba.mls.api.ApplicationProvider
import com.alibaba.mls.api.download.ModelDownloadManager
import com.alibaba.mnnllm.android.llm.GenerateProgressListener
import com.alibaba.mnnllm.android.llm.SanaSession
import com.alibaba.mnnllm.android.model.ModelTypeUtils
import com.alibaba.mnnllm.android.modelist.ModelListManager
import com.facebook.stetho.dumpapp.DumperContext
import com.facebook.stetho.dumpapp.DumperPlugin
import java.io.File
import java.io.PrintStream

/**
 * Stetho DumperPlugin for testing Sana diffusion model via command line.
 * 
 * Usage:
 *   dumpapp sana list                    - List installed Sana models
 *   dumpapp sana run <model_path> <prompt> [options]  - Run generation
 * 
 * Options:
 *   --output <path>     Output image path (default: /sdcard/sana_output.jpg)
 *   --input <path>      Input image for img2img mode (optional)
 *   --steps <n>         Number of inference steps (default: 5)
 *   --seed <n>          Random seed (default: 42)
 *   --cfg <true|false>  Use CFG mode (default: true)
 *   --cfg-scale <f>     CFG scale (default: 4.5)
 *   --width <n>         Output width (default: 512)
 *   --height <n>        Output height (default: 512)
 * 
 * Example:
 *   dumpapp sana list
 *   dumpapp sana run /data/local/tmp/sana_model "a cute cat"
 *   dumpapp sana run /data/local/tmp/sana_model "ghibli style" --input /sdcard/photo.jpg --cfg true
 */
class SanaDumperPlugin : DumperPlugin {

    override fun getName(): String = "sana"

    override fun dump(dumpContext: DumperContext) {
        val writer = dumpContext.stdout
        val args = dumpContext.argsAsList

        if (args.isEmpty()) {
            doUsage(writer)
            return
        }

        when (args[0]) {
            "list" -> doList(writer)
            "run" -> doRun(writer, args.drop(1))
            else -> doUsage(writer)
        }
    }

    private fun doList(writer: PrintStream) {
        writer.println("=== Installed Sana/Diffusion Models ===")
        writer.println()

        // Check common local paths
        val localPaths = listOf(
            "/data/local/tmp/sana_mnn_models_v2_distill",
            "/data/local/tmp/sana_model",
            "/sdcard/mnn_models/sana"
        )

        writer.println("[Local Paths]")
        for (path in localPaths) {
            val dir = File(path)
            if (dir.exists() && dir.isDirectory) {
                val hasLlm = File(dir, "llm/config.json").exists()
                val hasTransformer = File(dir, "transformer.mnn").exists()
                val status = if (hasLlm && hasTransformer) "VALID" else "INCOMPLETE"
                writer.println("  $path [$status]")
                if (hasLlm) writer.println("    - llm/config.json: OK")
                if (hasTransformer) writer.println("    - transformer.mnn: OK")
            }
        }
        writer.println()

        // Check downloaded models
        writer.println("[Downloaded Models]")
        val state = ModelListManager.modelListState.value
        if (state is ModelListManager.ModelListState.Success) {
            val diffusionModels = state.models.filter { 
                ModelTypeUtils.isDiffusionModel(it.modelItem.modelName ?: it.modelItem.modelId ?: "")
            }
            if (diffusionModels.isEmpty()) {
                writer.println("  No diffusion models found in ModelListManager")
            } else {
                diffusionModels.forEach { model ->
                    val modelId = model.modelItem.modelId
                    val isLocal = model.isLocal
                    val localPath = model.modelItem.localPath
                    writer.println("  - $modelId")
                    writer.println("    Local: $isLocal, Path: $localPath")
                }
            }
        } else {
            writer.println("  ModelListManager not ready: $state")
        }
        writer.println()

        // Check ModelDownloadManager
        writer.println("[ModelDownloadManager]")
        try {
            val downloadManager = ModelDownloadManager.getInstance(ApplicationProvider.get())
            val sanaModels = listOf(
                "HuggingFace/taobao-mnn/Sana-0.6B",
                "ModelScope/MNN/Sana-0.6B",
                "Modelers/MNN/Sana-0.6B"
            )
            for (modelId in sanaModels) {
                val downloadedFile = downloadManager.getDownloadedFile(modelId)
                if (downloadedFile != null && downloadedFile.exists()) {
                    writer.println("  $modelId -> ${downloadedFile.absolutePath}")
                }
            }
        } catch (e: Exception) {
            writer.println("  Error checking downloads: ${e.message}")
        }
    }

    private fun doRun(writer: PrintStream, args: List<String>) {
        if (args.size < 2) {
            writer.println("Error: Missing required arguments")
            writer.println("Usage: dumpapp sana run <model_path> <prompt> [options]")
            return
        }

        val modelPath = args[0]
        val prompt = args[1]

        // Parse options
        var outputPath = "/sdcard/sana_output.jpg"
        var inputPath = ""
        var steps = 5
        var seed = 42
        var useCfg = true
        var cfgScale = 4.5f
        var width = 512
        var height = 512

        var i = 2
        while (i < args.size) {
            when (args[i]) {
                "--output" -> { outputPath = args.getOrElse(i + 1) { outputPath }; i += 2 }
                "--input" -> { inputPath = args.getOrElse(i + 1) { inputPath }; i += 2 }
                "--steps" -> { steps = args.getOrElse(i + 1) { "$steps" }.toIntOrNull() ?: steps; i += 2 }
                "--seed" -> { seed = args.getOrElse(i + 1) { "$seed" }.toIntOrNull() ?: seed; i += 2 }
                "--cfg" -> { useCfg = args.getOrElse(i + 1) { "true" }.toBoolean(); i += 2 }
                "--cfg-scale" -> { cfgScale = args.getOrElse(i + 1) { "$cfgScale" }.toFloatOrNull() ?: cfgScale; i += 2 }
                "--width" -> { width = args.getOrElse(i + 1) { "$width" }.toIntOrNull() ?: width; i += 2 }
                "--height" -> { height = args.getOrElse(i + 1) { "$height" }.toIntOrNull() ?: height; i += 2 }
                else -> i++
            }
        }

        writer.println("=== Sana Generation ===")
        writer.println("Model: $modelPath")
        writer.println("Prompt: $prompt")
        writer.println("Mode: ${if (inputPath.isEmpty()) "text2img" else "img2img"}")
        if (inputPath.isNotEmpty()) writer.println("Input: $inputPath")
        writer.println("Output: $outputPath")
        writer.println("Size: ${width}x${height}")
        writer.println("Steps: $steps, Seed: $seed")
        writer.println("CFG: $useCfg, Scale: $cfgScale")
        writer.println()

        // Validate model path
        val modelDir = File(modelPath)
        if (!modelDir.exists()) {
            writer.println("ERROR: Model path does not exist: $modelPath")
            return
        }

        val llmConfig = File(modelDir, "llm/config.json")
        if (!llmConfig.exists()) {
            writer.println("ERROR: LLM config not found: ${llmConfig.absolutePath}")
            return
        }

        // Create config JSON
        val configJson = """
            {
                "diffusion_memory_mode": "0",
                "backend_type": "opencl",
                "image_width": $width,
                "image_height": $height,
                "grid_size": 1
            }
        """.trimIndent()

        writer.println("Initializing SanaSession...")
        writer.flush()

        try {
            // Use correct modelId format: "local/" + path
            val modelId = "local/$modelPath"
            val session = SanaSession(
                modelId = modelId,
                sessionId = System.currentTimeMillis().toString(),
                configPath = modelPath
            )
            session.load()

            writer.println("Session loaded, starting generation...")
            writer.flush()

            val params = hashMapOf<String, Any>(
                "output" to outputPath,
                "imageInput" to inputPath,
                "iterNum" to steps,
                "randomSeed" to seed,
                "useCfg" to useCfg,
                "cfgScale" to cfgScale
            )

            val progressListener = object : GenerateProgressListener {
                override fun onProgress(progress: String?): Boolean {
                    writer.println("Progress: $progress%")
                    writer.flush()
                    return true
                }
            }

            val startTime = System.currentTimeMillis()
            val result = session.generate(prompt, params, progressListener)
            val elapsed = System.currentTimeMillis() - startTime

            writer.println()
            writer.println("=== Result ===")
            writer.println("Time: ${elapsed}ms (${elapsed / 1000.0}s)")
            writer.println("Success: ${result["success"]}")
            if (result["error"] == true) {
                writer.println("Error: ${result["message"]}")
            } else {
                writer.println("Output saved to: $outputPath")
                val outputFile = File(outputPath)
                if (outputFile.exists()) {
                    writer.println("File size: ${outputFile.length()} bytes")
                }
            }

            session.release()
        } catch (e: Exception) {
            writer.println("ERROR: ${e.message}")
            e.printStackTrace(writer)
        }
    }

    private fun doUsage(writer: PrintStream) {
        writer.println("Sana Diffusion Model Tester")
        writer.println()
        writer.println("Usage: dumpapp sana <command> [args]")
        writer.println()
        writer.println("Commands:")
        writer.println("  list                              List installed Sana models")
        writer.println("  run <model_path> <prompt> [opts]  Run image generation")
        writer.println()
        writer.println("Options for 'run':")
        writer.println("  --output <path>     Output image path (default: /sdcard/sana_output.jpg)")
        writer.println("  --input <path>      Input image for img2img mode")
        writer.println("  --steps <n>         Inference steps (default: 5)")
        writer.println("  --seed <n>          Random seed (default: 42)")
        writer.println("  --cfg <true|false>  Use CFG mode (default: true)")
        writer.println("  --cfg-scale <f>     CFG scale (default: 4.5)")
        writer.println("  --width <n>         Output width (default: 512)")
        writer.println("  --height <n>        Output height (default: 512)")
        writer.println()
        writer.println("Examples:")
        writer.println("  dumpapp sana list")
        writer.println("  dumpapp sana run /data/local/tmp/sana_mnn_models_v2_distill \"a cute cat\"")
        writer.println("  dumpapp sana run /data/local/tmp/sana_model \"ghibli style\" --input /sdcard/photo.jpg")
    }
}
