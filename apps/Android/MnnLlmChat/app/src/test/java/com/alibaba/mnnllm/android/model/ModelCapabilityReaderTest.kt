package com.alibaba.mnnllm.android.model

import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import java.io.File
import java.nio.file.Files

// org.json needs the Android implementation, which Robolectric provides.
@RunWith(RobolectricTestRunner::class)
class ModelCapabilityReaderTest {
    private lateinit var dir: File

    @Before
    fun setUp() {
        dir = Files.createTempDirectory("capability").toFile()
    }

    @After
    fun tearDown() {
        dir.deleteRecursively()
    }

    private fun write(json: String, mtime: Long? = null) = File(dir, "llm_config.json").apply {
        writeText(json)
        mtime?.let { setLastModified(it) }
    }

    @Test
    fun `declared flags are read`() {
        write("""{"is_visual": true, "is_audio": false}""")
        assertEquals(ModelCapabilities(true, false), ModelCapabilityReader.read(dir))
        assertEquals(listOf("Vision"), ModelCapabilityReader.modalityTags(dir))
    }

    @Test
    fun `absent, null or non-boolean fields are unknown`() {
        write("""{"hidden_size": 1024, "is_visual": null, "is_audio": "yes"}""")
        assertEquals(ModelCapabilities(null, null), ModelCapabilityReader.read(dir))
    }

    @Test
    fun `missing or corrupt config is unknown`() {
        assertEquals(ModelCapabilities.UNKNOWN, ModelCapabilityReader.read(dir))
        write("""{"is_visual": tr""")
        assertEquals(ModelCapabilities.UNKNOWN, ModelCapabilityReader.read(dir))
        assertEquals(emptyList<String>(), ModelCapabilityReader.modalityTags(dir))
    }

    @Test
    fun `an updated config is read again`() {
        write("""{"is_visual": false}""", mtime = 1_000_000L)
        assertEquals(false, ModelCapabilityReader.read(dir).isVisual)
        write("""{"is_visual": true}""", mtime = 2_000_000L)
        assertEquals(true, ModelCapabilityReader.read(dir).isVisual)
    }

    @Test
    fun `config path resolves to its directory`() {
        write("""{"is_audio": true}""")
        File(dir, "config.json").writeText("{}")
        assertEquals(true, ModelCapabilityReader.readForConfigPath(File(dir, "config.json").path).isAudio)
        assertNull(ModelCapabilityReader.readForConfigPath(null).isAudio)
    }
}
