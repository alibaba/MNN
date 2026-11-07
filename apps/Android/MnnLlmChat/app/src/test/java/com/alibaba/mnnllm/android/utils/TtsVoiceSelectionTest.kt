package com.alibaba.mnnllm.android.utils

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.io.File
import java.nio.file.Files

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [28])
class TtsVoiceSelectionTest {
    private lateinit var root: File
    private val context get() = ApplicationProvider.getApplicationContext<Context>()

    @Before fun setup() {
        root = Files.createTempDirectory("tts-voices").toFile()
        File(root, "config.json").writeText("""{"model_type":"supertonic"}""")
        TtsVoiceSelection.saveVoice(context, "model-a", null)
        TtsVoiceSelection.saveVoice(context, "model-b", null)
    }

    @After fun cleanup() { root.deleteRecursively() }

    private fun asset(path: String) {
        File(root, path).apply { parentFile!!.mkdirs(); writeText("{}") }
    }

    @Test fun findsOnlyNativeSupportedIdsInBothLayouts() {
        asset("mnn_models/voice_styles/F2.json")
        asset("voice_styles/M1.json")
        asset("voice_styles/custom.json")
        assertEquals(listOf("M1", "F2"), TtsVoiceSelection.availableVoices(root.path))
    }

    @Test fun unsupportedMissingAndInvalidConfigHaveNoChoices() {
        asset("voice_styles/M1.json")
        File(root, "config.json").writeText("""{"model_type":"bertvits"}""")
        assertTrue(TtsVoiceSelection.availableVoices(root.path).isEmpty())
        File(root, "config.json").writeText("invalid")
        assertTrue(TtsVoiceSelection.availableVoices(root.path).isEmpty())
        File(root, "config.json").delete()
        assertTrue(TtsVoiceSelection.availableVoices(root.path).isEmpty())
    }

    @Test fun preferenceIsPerModelAndDefaultClearsOverride() {
        asset("voice_styles/F1.json")
        TtsVoiceSelection.saveVoice(context, "model-a", "F1")
        assertEquals("F1", TtsVoiceSelection.selectedVoice(context, "model-a", root.path))
        assertNull(TtsVoiceSelection.selectedVoice(context, "model-b", root.path))
        assertNull(TtsVoiceSelection.selectedVoice(context, null, root.path))
        TtsVoiceSelection.saveVoice(context, "model-a", null)
        assertNull(TtsVoiceSelection.savedVoice(context, "model-a"))
    }

    @Test fun deletedOrUnknownVoiceDoesNotReachNativeSetter() {
        asset("voice_styles/F1.json")
        TtsVoiceSelection.saveVoice(context, "model-a", "F1")
        File(root, "voice_styles/F1.json").delete()
        assertNull(TtsVoiceSelection.selectedVoice(context, "model-a", root.path))
        TtsVoiceSelection.saveVoice(context, "model-a", "../../other")
        assertNull(TtsVoiceSelection.selectedVoice(context, "model-a", root.path))
    }
}
