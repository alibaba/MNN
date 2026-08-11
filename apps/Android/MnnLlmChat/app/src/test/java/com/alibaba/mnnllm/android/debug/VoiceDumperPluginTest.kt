package com.alibaba.mnnllm.android.debug

import android.content.Context
import com.alibaba.mnnllm.android.utils.VoiceModelPathUtils
import io.mockk.coEvery
import io.mockk.every
import io.mockk.just
import io.mockk.mockk
import io.mockk.mockkObject
import io.mockk.runs
import io.mockk.unmockkAll
import io.mockk.verifyOrder
import org.junit.After
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.PrintStream
import kotlinx.coroutines.runBlocking

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [28])
class VoiceDumperPluginTest {

    @After
    fun tearDown() {
        unmockkAll()
    }

    @Test
    fun `status prints language and sample rate`() {
        val context = mockk<Context>(relaxed = true)
        mockkObject(VoiceModelPathUtils)
        every { VoiceModelPathUtils.checkVoiceModelsStatus(context) } returns (true to "ok")
        every { VoiceModelPathUtils.getTtsModelPath(context) } returns "/tts/model"
        every { VoiceModelPathUtils.getAsrModelPath(context) } returns "/asr/model"
        every { VoiceModelPathUtils.getTtsLanguage(context) } returns "zh"
        every { VoiceModelPathUtils.getTtsSampleRate("/tts/model", any()) } returns 24000

        val output = ByteArrayOutputStream()
        val plugin = VoiceDumperPlugin(contextProvider = { context })

        plugin.execute(listOf("status"), PrintStream(output))

        val text = output.toString()
        assertTrue(text.contains("TTS_MODEL_PATH=/tts/model"))
        assertTrue(text.contains("TTS_LANGUAGE=zh"))
        assertTrue(text.contains("TTS_SAMPLE_RATE=24000"))
    }

    @Test
    fun `tts init sets language before init`() {
        val context = mockk<Context>(relaxed = true)
        val ttsService = mockk<TtsClient>()
        mockkObject(VoiceModelPathUtils)
        every { VoiceModelPathUtils.getTtsModelPath(context) } returns "/tts/model"
        every { VoiceModelPathUtils.getTtsLanguage(context) } returns "zh"
        every { ttsService.setLanguage("zh") } just runs
        coEvery { ttsService.init("/tts/model") } returns true

        val output = ByteArrayOutputStream()
        val plugin = VoiceDumperPlugin(
            contextProvider = { context },
            ttsClientFactory = { ttsService }
        )

        plugin.execute(listOf("tts", "init"), PrintStream(output))

        verifyOrder {
            ttsService.setLanguage("zh")
            runBlocking { ttsService.init("/tts/model") }
        }
    }

    @Test
    fun `tts init accepts explicit model path override`() {
        val context = mockk<Context>(relaxed = true)
        val ttsService = mockk<TtsClient>()
        mockkObject(VoiceModelPathUtils)
        every { VoiceModelPathUtils.getTtsModelPath(context) } returns "/tts/default"
        every { VoiceModelPathUtils.getTtsLanguage(context) } returns "zh"
        every { ttsService.setLanguage("zh") } just runs
        coEvery { ttsService.init("/tmp/demo-model") } returns true

        val output = ByteArrayOutputStream()
        val plugin = VoiceDumperPlugin(
            contextProvider = { context },
            ttsClientFactory = { ttsService }
        )

        plugin.execute(listOf("tts", "init", "/tmp/demo-model"), PrintStream(output))

        val text = output.toString()
        assertTrue(text.contains("TTS_MODEL_PATH=/tmp/demo-model"))
        verifyOrder {
            ttsService.setLanguage("zh")
            runBlocking { ttsService.init("/tmp/demo-model") }
        }
    }

    @Test
    fun `tts test prints roundtrip diagnostics`() {
        val context = mockk<Context>(relaxed = true)
        val ttsService = mockk<TtsClient>()
        val roundTripEvaluator = mockk<TtsRoundTripEvaluator>()
        mockkObject(VoiceModelPathUtils)
        every { VoiceModelPathUtils.getTtsModelPath(context) } returns "/tts/model"
        every { VoiceModelPathUtils.getTtsSampleRate("/tts/model", any()) } returns 44100
        coEvery { ttsService.waitForInitComplete() } returns true
        every { ttsService.process("你好，世界。", 0) } returns ShortArray(44100) { 1 }
        every {
            roundTripEvaluator.evaluate(context, any(), 44100, "你好，世界。")
        } returns TtsRoundTripResult(
            recognizedText = "你好世界",
            hasChinese = true,
            similarity = 1.0,
            status = "PASS"
        )

        val output = ByteArrayOutputStream()
        val plugin = VoiceDumperPlugin(
            contextProvider = { context },
            ttsClientFactory = { ttsService },
            roundTripEvaluatorFactory = { roundTripEvaluator }
        )

        val initializedField = VoiceDumperPlugin::class.java.getDeclaredField("isTtsInitialized")
        initializedField.isAccessible = true
        initializedField.setBoolean(plugin, true)

        val serviceField = VoiceDumperPlugin::class.java.getDeclaredField("ttsService")
        serviceField.isAccessible = true
        serviceField.set(plugin, ttsService)

        plugin.execute(listOf("tts", "test", "你好，世界。"), PrintStream(output))

        val text = output.toString()
        assertTrue(text.contains("TTS_TEST=SUCCESS"))
        assertTrue(text.contains("TTS_ROUNDTRIP_TEXT=你好世界"))
        assertTrue(text.contains("TTS_ROUNDTRIP_HAS_CHINESE=true"))
        assertTrue(text.contains("TTS_ROUNDTRIP_STATUS=PASS"))
    }

    @Test
    fun `tts test exports wav artifact and uses initialized model path sample rate`() {
        val tempDir = createTempDir(prefix = "voice-dumper-test")
        val exportedFile = File(tempDir, "tts-output.wav")
        val context = mockk<Context>(relaxed = true)
        val ttsService = mockk<TtsClient>()
        val roundTripEvaluator = mockk<TtsRoundTripEvaluator>()
        val artifactWriter = mockk<TtsAudioArtifactWriter>()
        mockkObject(VoiceModelPathUtils)
        every { context.cacheDir } returns tempDir
        every { VoiceModelPathUtils.getTtsModelPath(context) } returns "/tts/default"
        every { VoiceModelPathUtils.getTtsSampleRate("/tts/default", any()) } returns 44100
        every { VoiceModelPathUtils.getTtsSampleRate("/tmp/demo-model", any()) } returns 22050
        coEvery { ttsService.waitForInitComplete() } returns true
        every { ttsService.process("你好，世界。", 0) } returns ShortArray(22050) { 1 }
        every {
            artifactWriter.write(context, any(), 22050)
        } answers {
            exportedFile.writeBytes(byteArrayOf(1, 2, 3, 4))
            exportedFile.absolutePath
        }
        every {
            roundTripEvaluator.evaluate(context, any(), 22050, "你好，世界。")
        } returns TtsRoundTripResult(
            recognizedText = "你好世界",
            hasChinese = true,
            similarity = 1.0,
            status = "PASS"
        )

        val output = ByteArrayOutputStream()
        val plugin = VoiceDumperPlugin(
            contextProvider = { context },
            ttsClientFactory = { ttsService },
            roundTripEvaluatorFactory = { roundTripEvaluator },
            audioArtifactWriter = artifactWriter
        )

        val initializedField = VoiceDumperPlugin::class.java.getDeclaredField("isTtsInitialized")
        initializedField.isAccessible = true
        initializedField.setBoolean(plugin, true)

        val serviceField = VoiceDumperPlugin::class.java.getDeclaredField("ttsService")
        serviceField.isAccessible = true
        serviceField.set(plugin, ttsService)

        val modelPathField = VoiceDumperPlugin::class.java.getDeclaredField("initializedTtsModelPath")
        modelPathField.isAccessible = true
        modelPathField.set(plugin, "/tmp/demo-model")

        plugin.execute(listOf("tts", "test", "你好，世界。"), PrintStream(output))

        val text = output.toString()
        assertTrue(text.contains("TTS_SAMPLE_RATE=22050"))
        assertTrue(text.contains("TTS_AUDIO_WAV_PATH=${exportedFile.absolutePath}"))
    }
}
