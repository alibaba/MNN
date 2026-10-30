package com.alibaba.mnnllm.android.modelist

import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import org.json.JSONObject
import java.io.File
import java.io.IOException
import java.io.InputStream
import java.util.UUID

/** Copies a prepared MNN bundle without exposing a partially copied model to the list. */
class LocalModelImporter(
    private val root: File,
    private val availableBytes: () -> Long = { root.usableSpace }
) {
    interface Source {
        val name: String
        val isDirectory: Boolean
        fun children(): List<Source>
        fun open(): InputStream
    }

    suspend fun import(source: Source): File = importLock.withLock {
        requireName(source.name)
        if (source.name.startsWith(".")) throw IOException("Choose a non-hidden model folder")
        if (!source.isDirectory) throw IOException("Select a model folder containing config.json")
        if (!root.isDirectory && !root.mkdirs()) throw IOException("Cannot create model storage")
        // Only this importer owns these hidden directories. A killed process cannot publish one.
        root.listFiles()?.filter { it.name.startsWith(STAGING_PREFIX) }?.forEach {
            if (!it.deleteRecursively()) throw IOException("Cannot clean an interrupted import")
        }
        val destination = File(root, source.name)
        if (destination.exists()) throw IOException("A model named ${source.name} already exists")
        val staging = File(root, STAGING_PREFIX + UUID.randomUUID())
        if (!staging.mkdir()) throw IOException("Cannot create import staging folder")
        try {
            copyDirectory(source, staging, 0)
            validateConfig(staging)
            currentCoroutineContext().ensureActive()
            // Same-filesystem rename is the publication point; never merge into an existing model.
            if (destination.exists() || !staging.renameTo(destination)) {
                throw IOException("Cannot publish imported model; check storage and duplicate names")
            }
            destination
        } finally {
            staging.deleteRecursively()
        }
    }

    private suspend fun copyDirectory(source: Source, target: File, depth: Int) {
        currentCoroutineContext().ensureActive()
        if (depth > 64) throw IOException("Model folder nesting is too deep")
        val names = mutableSetOf<String>()
        for (child in source.children()) {
            currentCoroutineContext().ensureActive()
            requireName(child.name)
            if (!names.add(child.name)) throw IOException("Duplicate file name: ${child.name}")
            val output = File(target, child.name)
            if (child.isDirectory) {
                if (!output.mkdir()) throw IOException("Cannot create ${child.name}")
                copyDirectory(child, output, depth + 1)
            } else {
                child.open().use { input ->
                    output.outputStream().use { stream ->
                        val buffer = ByteArray(64 * 1024)
                        while (true) {
                            currentCoroutineContext().ensureActive()
                            val count = input.read(buffer)
                            if (count < 0) break
                            if (count == 0) continue
                            if (availableBytes() < count) throw IOException("Not enough free storage to import this model")
                            stream.write(buffer, 0, count)
                        }
                    }
                }
            }
        }
    }

    private fun validateConfig(directory: File) {
        val configFile = File(directory, "config.json")
        if (!configFile.isFile || configFile.length() > 1024 * 1024) {
            throw IOException("Select a model folder with a valid config.json at its root")
        }
        val config = try {
            JSONObject(configFile.readText())
        } catch (e: Exception) {
            throw IOException("Invalid config.json", e)
        }
        requireModelFile(directory, config, "llm_model", "llm.mnn")
        if (config.has("llm_weight")) requireModelFile(directory, config, "llm_weight", null)
    }

    private fun requireModelFile(directory: File, config: JSONObject, key: String, default: String?) {
        val value = if (config.has(key)) config.opt(key) else default
        if (value !is String || value.isBlank()) throw IOException("Invalid $key in config.json")
        val file = File(directory, value)
        if (File(value).isAbsolute || !file.canonicalPath.startsWith(directory.canonicalPath + File.separator) ||
            !file.isFile || file.length() == 0L) {
            throw IOException("Missing or invalid $key file: $value")
        }
    }

    private fun requireName(name: String) {
        if (name.isBlank() || (name == "." || name == "..") || name.contains('/') || name.contains('\\') || name.contains('\u0000')) {
            throw IOException("Unsupported file or folder name: $name")
        }
    }

    companion object {
        private const val STAGING_PREFIX = ".import-"
        private val importLock = Mutex()
    }
}
