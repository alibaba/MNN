package com.alibaba.mnntalk.model

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class VoiceModelDirectoryResolverTest {
    @get:Rule
    val temporaryFolder = TemporaryFolder()

    @Test
    fun resolvesFlatMnnBundle() {
        val root = temporaryFolder.newFolder("MNN")
        createModel(root, VoiceModelDirectoryResolver.LLM_DIRECTORY)
        createModel(root, VoiceModelDirectoryResolver.ASR_DIRECTORY)
        createModel(root, VoiceModelDirectoryResolver.TTS_DIRECTORY)

        val paths = VoiceModelDirectoryResolver.resolve(root)

        assertNotNull(paths)
        assertEquals(
            File(root, "${VoiceModelDirectoryResolver.LLM_DIRECTORY}/config.json"),
            paths?.llmConfig
        )
        assertEquals(VoiceModelSource.DEVELOPER_DIRECTORY, paths?.source)
        assertEquals(root, paths?.rootDirectory)
    }

    @Test
    fun resolvesModelScopeLayout() {
        val root = temporaryFolder.newFolder("models")
        val mnnRoot = File(root, "ModelScope/MNN")
        createModel(mnnRoot, VoiceModelDirectoryResolver.LLM_DIRECTORY)
        createModel(mnnRoot, VoiceModelDirectoryResolver.ASR_DIRECTORY)
        createModel(mnnRoot, VoiceModelDirectoryResolver.TTS_DIRECTORY)

        assertNotNull(VoiceModelDirectoryResolver.resolve(root))
    }

    @Test
    fun resolvesDownloadedAsrWithoutConfigJson() {
        val root = temporaryFolder.newFolder("downloaded")
        createModel(root, VoiceModelDirectoryResolver.LLM_DIRECTORY)
        createFallbackAsrModel(root)
        createModel(root, VoiceModelDirectoryResolver.TTS_DIRECTORY)

        assertNotNull(VoiceModelDirectoryResolver.resolve(root))
    }

    @Test
    fun rejectsIncompleteBundle() {
        val root = temporaryFolder.newFolder("incomplete")
        createModel(root, VoiceModelDirectoryResolver.LLM_DIRECTORY)
        createModel(root, VoiceModelDirectoryResolver.ASR_DIRECTORY)

        assertNull(VoiceModelDirectoryResolver.resolve(root))
    }

    private fun createModel(root: File, directoryName: String) {
        File(root, directoryName).apply {
            mkdirs()
            File(this, "config.json").writeText("{}")
        }
    }

    private fun createFallbackAsrModel(root: File) {
        File(root, VoiceModelDirectoryResolver.ASR_DIRECTORY).apply {
            mkdirs()
            File(this, "encoder-epoch-99-avg-1.int8.mnn").writeText("")
            File(this, "decoder-epoch-99-avg-1.int8.mnn").writeText("")
            File(this, "joiner-epoch-99-avg-1.int8.mnn").writeText("")
            File(this, "tokens.txt").writeText("")
        }
    }
}
