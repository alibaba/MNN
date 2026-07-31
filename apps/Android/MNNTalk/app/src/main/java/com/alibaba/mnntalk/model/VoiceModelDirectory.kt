package com.alibaba.mnntalk.model

import java.io.File

enum class VoiceModelSource {
    DEVELOPER_DIRECTORY,
    DOWNLOADER
}

data class VoiceModelPaths(
    val llmConfig: File,
    val asrDirectory: File,
    val ttsDirectory: File,
    val source: VoiceModelSource,
    val rootDirectory: File? = null
)

object VoiceModelDirectoryResolver {
    const val LLM_DIRECTORY = "Qwen3-0.6B-MNN"
    const val ASR_DIRECTORY =
        "sherpa-mnn-streaming-zipformer-bilingual-zh-en-2023-02-20"
    const val TTS_DIRECTORY = "bert-vits2-MNN"

    private val directoryPrefixes = listOf("", "MNN", "ModelScope/MNN")
    private val fallbackAsrFiles = listOf(
        "encoder-epoch-99-avg-1.int8.mnn",
        "decoder-epoch-99-avg-1.int8.mnn",
        "joiner-epoch-99-avg-1.int8.mnn",
        "tokens.txt"
    )

    fun resolve(rootDirectory: File): VoiceModelPaths? {
        val llmDirectory = findModelDirectory(rootDirectory, LLM_DIRECTORY) ?: return null
        val asrDirectory = findModelDirectory(
            rootDirectory,
            ASR_DIRECTORY,
            ::isAsrModelDirectory
        ) ?: return null
        val ttsDirectory = findModelDirectory(rootDirectory, TTS_DIRECTORY) ?: return null
        return resolveDirectories(
            llmDirectory = llmDirectory,
            asrDirectory = asrDirectory,
            ttsDirectory = ttsDirectory,
            source = VoiceModelSource.DEVELOPER_DIRECTORY,
            rootDirectory = rootDirectory
        )
    }

    fun resolveDirectories(
        llmDirectory: File,
        asrDirectory: File,
        ttsDirectory: File,
        source: VoiceModelSource,
        rootDirectory: File? = null
    ): VoiceModelPaths? {
        val llmConfig = File(llmDirectory, "config.json")
        val ttsConfig = File(ttsDirectory, "config.json")
        if (!llmConfig.isFile || !isAsrModelDirectory(asrDirectory) || !ttsConfig.isFile) {
            return null
        }
        return VoiceModelPaths(
            llmConfig = llmConfig,
            asrDirectory = asrDirectory,
            ttsDirectory = ttsDirectory,
            source = source,
            rootDirectory = rootDirectory
        )
    }

    private fun findModelDirectory(
        rootDirectory: File,
        directoryName: String,
        isValid: (File) -> Boolean = { File(it, "config.json").isFile }
    ): File? {
        return directoryPrefixes
            .asSequence()
            .map { prefix ->
                if (prefix.isEmpty()) {
                    File(rootDirectory, directoryName)
                } else {
                    File(rootDirectory, "$prefix/$directoryName")
                }
            }
            .firstOrNull { directory ->
                directory.isDirectory && isValid(directory)
            }
    }

    private fun isAsrModelDirectory(directory: File): Boolean {
        if (!directory.isDirectory) return false
        if (File(directory, "config.json").isFile) return true
        return fallbackAsrFiles.all { File(directory, it).isFile }
    }
}
