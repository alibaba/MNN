package com.alibaba.mnnllm.android.modelist

import android.util.Log
import android.content.Context
import com.alibaba.mnnllm.android.MnnLlmApplication
import com.alibaba.mls.api.ModelItem
import java.io.File

object LocalModelsProvider {
    fun importedModelsDir(context: Context): File = File(context.filesDir, "imported_models")

    /** Only app-owned imports may be removed here; adb and downloaded models keep their existing paths. */
    fun importedModelDirectory(context: Context, modelId: String): File? {
        if (!modelId.startsWith("local/")) return null
        val path = File(modelId.removePrefix("local/")).canonicalFile
        val root = importedModelsDir(context).canonicalFile
        return path.takeIf { it.parentFile == root && !it.name.startsWith(".") }
    }

    fun getLocalModels(): MutableList<ModelItem> = getLocalModels(MnnLlmApplication.getInstance())

    fun getLocalModels(context: Context): MutableList<ModelItem> {
        val result = mutableListOf<ModelItem>()
        val directories = listOf(File("/data/local/tmp/mnn_models/"), importedModelsDir(context))
        for (modelsDir in directories) {
            try {
                modelsDir.listFiles()?.forEach { modelDir ->
                    if (!modelDir.name.startsWith(".") && modelDir.isDirectory && File(modelDir, "config.json").isFile) {
                        val modelPath = modelDir.absolutePath
                        result.add(ModelItem.fromLocalModel("local/$modelPath", modelPath))
                    }
                }
            } catch (e: Exception) {
                Log.e("LocalModelsProvider", "Failed to load models from $modelsDir", e)
            }
        }
        return result
    }
}
