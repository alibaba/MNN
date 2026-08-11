package com.alibaba.mnnllm.android.utils

import android.content.Context
import android.content.res.Configuration
import androidx.test.core.app.ApplicationProvider
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.io.File
import java.nio.file.Files
import java.util.Locale

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [28])
class VoiceModelPathUtilsTest {

    private fun createContextWithLocale(locale: Locale): Context {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val config = Configuration(context.resources.configuration)
        config.setLocale(locale)
        return context.createConfigurationContext(config)
    }

    @Test
    fun getTtsSampleRate_readsSampleRateFromConfig() {
        val dir = Files.createTempDirectory("tts-sample-rate").toFile()
        try {
            File(dir, "config.json").writeText("""{"sample_rate":24000}""")

            val sampleRate = VoiceModelPathUtils.getTtsSampleRate(dir.absolutePath)

            assertEquals(24000, sampleRate)
        } finally {
            dir.deleteRecursively()
        }
    }

    @Test
    fun getTtsSampleRate_fallsBackWhenConfigMissing() {
        val dir = Files.createTempDirectory("tts-sample-rate-missing").toFile()
        try {
            val sampleRate = VoiceModelPathUtils.getTtsSampleRate(dir.absolutePath)

            assertEquals(VoiceModelPathUtils.DEFAULT_TTS_SAMPLE_RATE, sampleRate)
        } finally {
            dir.deleteRecursively()
        }
    }

    @Test
    fun getTtsSampleRate_fallsBackWhenConfigInvalid() {
        val dir = Files.createTempDirectory("tts-sample-rate-invalid").toFile()
        try {
            File(dir, "config.json").writeText("""{"sample_rate":"oops"}""")

            val sampleRate = VoiceModelPathUtils.getTtsSampleRate(dir.absolutePath)

            assertEquals(VoiceModelPathUtils.DEFAULT_TTS_SAMPLE_RATE, sampleRate)
        } finally {
            dir.deleteRecursively()
        }
    }

    @Test
    fun getTtsLanguage_returnsZhForChineseLocale() {
        val context = createContextWithLocale(Locale.SIMPLIFIED_CHINESE)

        val language = VoiceModelPathUtils.getTtsLanguage(context)

        assertEquals("zh", language)
    }

    @Test
    fun getTtsLanguage_returnsEnForEnglishLocale() {
        val context = createContextWithLocale(Locale.US)

        val language = VoiceModelPathUtils.getTtsLanguage(context)

        assertEquals("en", language)
    }
}
