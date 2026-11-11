package com.alibaba.mnnllm.android.modelist

import android.content.Context
import android.util.Log
import com.alibaba.mls.api.ApplicationProvider
import com.alibaba.mls.api.ModelItem
import com.alibaba.mnnllm.android.model.ModelCapabilityReader
import java.io.File

/**
 * Models that already sit on the device's file system. Three roots are scanned, each only when it
 * exists, because a missing root is the normal case:
 *
 * - `<external files>/models` (`Android/data/<pkg>/files/models`): any file manager can write here
 *   with no runtime permission and no adb, and the native runtime can open it by path, so a model
 *   folder dropped in is usable as-is.
 * - `<files>/.mnnlocalmodels`: where [LocalModelImporter] puts a folder chosen through the storage
 *   access framework picker. That picker only yields a content URI, which MNN's file IO cannot use,
 *   so the pick has to be copied. Kept out of `.mnnmodels` on purpose: that tree is owned by the
 *   downloader, which maps every path in it to a market model id.
 * - `/data/local/tmp/mnn_models`: the long-standing adb push location, kept for development.
 */
object LocalModelsProvider {
    private const val TAG = "LocalModelsProvider"
    const val ADB_MODELS_DIR = "/data/local/tmp/mnn_models/"
    const val IMPORTED_DIR_NAME = ".mnnlocalmodels"
    const val DROP_IN_DIR_NAME = "models"
    private const val ENTRY_CONFIG = "config.json"

    fun importedModelsDir(context: Context): File = File(context.filesDir, IMPORTED_DIR_NAME)

    fun dropInModelsDir(context: Context): File? = context.getExternalFilesDir(DROP_IN_DIR_NAME)

    /**
     * you can add ModelItem.fromLocalModel("Qwen-Omni-7B", "/data/local/tmp/omni_test/model")
     * to load local models
     */
    fun getLocalModels(): MutableList<ModelItem> {
        val roots = mutableListOf<File>()
        val context = appContext()
        if (context != null) {
            dropInModelsDir(context)?.let { roots.add(it) }
            roots.add(importedModelsDir(context))
        } else {
            Log.w(TAG, "No application context, scanning $ADB_MODELS_DIR only")
        }
        roots.add(File(ADB_MODELS_DIR))

        val result = mutableListOf<ModelItem>()
        roots.forEach { root ->
            try {
                result.addAll(modelsIn(root))
            } catch (e: Exception) {
                Log.e(TAG, "Failed to scan ${root.absolutePath}", e)
            }
        }
        return result
    }

    /**
     * The model folders under [root]: every sub-folder holding a config.json, plus [root] itself
     * when it is a model folder, which is what a picker aimed straight at a model yields.
     */
    fun modelsIn(root: File): List<ModelItem> {
        if (!root.isDirectory) {
            return emptyList()
        }
        val modelDirs = mutableListOf<File>()
        if (File(root, ENTRY_CONFIG).isFile) {
            modelDirs.add(root)
        }
        root.listFiles()?.forEach { child ->
            if (child.isDirectory && File(child, ENTRY_CONFIG).isFile) {
                modelDirs.add(child)
            }
        }
        return modelDirs.map { modelDir ->
            val modelPath = modelDir.absolutePath
            val item = ModelItem.fromLocalModel("local/$modelPath", modelPath)
            // A local model has no market metadata, so its own llm_config.json says what it accepts.
            item.tags = ModelCapabilityReader.modalityTags(modelDir)
            item
        }
    }

    private fun appContext(): Context? {
        return try {
            ApplicationProvider.get()
        } catch (e: Exception) {
            null
        }
    }
}
