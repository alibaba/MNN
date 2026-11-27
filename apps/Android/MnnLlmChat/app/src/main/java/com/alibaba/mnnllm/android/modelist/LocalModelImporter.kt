package com.alibaba.mnnllm.android.modelist

import android.content.Context
import android.net.Uri
import android.util.Log
import androidx.documentfile.provider.DocumentFile
import java.io.File

/**
 * Materialises a folder the user picked through the storage access framework into app-private
 * storage. The picker hands out a content URI and MNN loads weights with plain file IO, so the
 * bytes have to be copied before the model can be opened; [LocalModelsProvider.dropInModelsDir] is
 * the copy-free alternative for users who have a file manager.
 *
 * Importing is by folder, not by persisted URI: once copied the model lives in app storage and
 * survives a reboot or a revoked permission. Picking the same folder again replaces the copy.
 */
object LocalModelImporter {
    private const val TAG = "LocalModelImporter"
    private const val BUFFER_SIZE = 1 shl 16
    private const val ENTRY_CONFIG = "config.json"

    data class Progress(val modelName: String, val fileName: String, val copiedBytes: Long, val totalBytes: Long)

    data class Result(val importedPaths: List<String>, val failedNames: List<String>) {
        val isEmpty: Boolean get() = importedPaths.isEmpty() && failedNames.isEmpty()
    }

    /**
     * The model folders inside [treeUri]: the picked folder itself when it holds a config.json,
     * otherwise its sub-folders that do. A folder without config.json anywhere is not a model, and
     * reporting it as one would only produce a launch failure later on.
     */
    fun findModelDirs(context: Context, treeUri: Uri): List<DocumentFile> {
        val root = DocumentFile.fromTreeUri(context, treeUri)
        if (root == null || !root.isDirectory) {
            return emptyList()
        }
        if (root.isModelDir()) {
            return listOf(root)
        }
        return root.listFiles().filter { it.isDirectory && it.isModelDir() }
    }

    fun import(context: Context, treeUri: Uri, onProgress: (Progress) -> Unit): Result {
        val destRoot = LocalModelsProvider.importedModelsDir(context)
        val imported = mutableListOf<String>()
        val failed = mutableListOf<String>()
        findModelDirs(context, treeUri).forEach { modelDir ->
            val name = modelDir.name
            if (name.isNullOrEmpty()) {
                failed.add(modelDir.uri.toString())
                return@forEach
            }
            val dest = File(destRoot, name)
            dest.deleteRecursively()
            val copied = copyTree(context, modelDir, dest, name, onProgress)
            if (copied && File(dest, ENTRY_CONFIG).isFile) {
                imported.add(dest.absolutePath)
            } else {
                dest.deleteRecursively()
                failed.add(name)
            }
        }
        return Result(imported, failed)
    }

    private fun DocumentFile.isModelDir(): Boolean = findFile(ENTRY_CONFIG)?.isFile == true

    private fun copyTree(
        context: Context,
        source: DocumentFile,
        dest: File,
        modelName: String,
        onProgress: (Progress) -> Unit
    ): Boolean {
        if (source.isFile) {
            return copyFile(context, source, dest, modelName, onProgress)
        }
        if (!dest.exists() && !dest.mkdirs()) {
            Log.e(TAG, "Cannot create ${dest.absolutePath}")
            return false
        }
        var ok = true
        source.listFiles().forEach { child ->
            val name = child.name
            if (name.isNullOrEmpty()) {
                return@forEach
            }
            if (!copyTree(context, child, File(dest, name), modelName, onProgress)) {
                ok = false
            }
        }
        return ok
    }

    private fun copyFile(
        context: Context,
        source: DocumentFile,
        dest: File,
        modelName: String,
        onProgress: (Progress) -> Unit
    ): Boolean {
        val total = source.length()
        return try {
            dest.parentFile?.mkdirs()
            val input = context.contentResolver.openInputStream(source.uri)
            if (input == null) {
                Log.e(TAG, "Cannot open ${source.name} for reading")
                return false
            }
            input.use { from ->
                dest.outputStream().use { to ->
                    val buffer = ByteArray(BUFFER_SIZE)
                    var copied = 0L
                    while (true) {
                        val read = from.read(buffer)
                        if (read <= 0) {
                            break
                        }
                        to.write(buffer, 0, read)
                        copied += read
                        onProgress(Progress(modelName, dest.name, copied, total))
                    }
                }
            }
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to copy ${source.name} to ${dest.absolutePath}", e)
            dest.delete()
            false
        }
    }
}
