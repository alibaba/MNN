package com.alibaba.mnnllm.android.llm

import android.app.Application
import androidx.test.core.app.ApplicationProvider
import com.alibaba.mls.api.ApplicationProvider as MlsApplicationProvider
import com.alibaba.mnnllm.android.modelsettings.ModelConfig
import com.google.gson.Gson
import io.mockk.every
import io.mockk.mockkStatic
import io.mockk.unmockkAll
import org.junit.After
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.io.File

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [28])
class DiffusionLoadConfigResolverTest {

    private lateinit var application: Application

    @Before
    fun setUp() {
        application = ApplicationProvider.getApplicationContext()
        mockkStatic(MlsApplicationProvider::class)
        every { MlsApplicationProvider.get() } returns application
    }

    @After
    fun tearDown() {
        File(application.filesDir, "configs").deleteRecursively()
        unmockkAll()
    }

    @Test
    fun buildExtraConfigJson_prefersModelLevelOverride_overLegacyPreference() {
        val modelId = "local/${createModelDir("override").absolutePath}"
        val configPath = ModelConfig.getDefaultConfigFile(modelId)!!
        val extraConfigPath = ModelConfig.getExtraConfigFile(modelId)

        ModelConfig.saveConfig(
            extraConfigPath,
            ModelConfig.defaultConfig.deepCopy().apply {
                diffusionMemoryMode = "2"
            }
        )

        val extraConfigJson = DiffusionLoadConfigResolver.buildExtraConfigJson(modelId, configPath)

        assertTrue(extraConfigJson.contains("\"diffusion_memory_mode\":\"2\""))
    }

    @Test
    fun buildExtraConfigJson_usesBaseConfig_whenNoOverrideExists() {
        val modelDir = createModelDir("base-only")
        val configPath = File(modelDir, "config.json").absolutePath
        val baseConfig = ModelConfig.defaultConfig.deepCopy().apply {
            diffusionMemoryMode = "1"
        }
        File(configPath).writeText(
            Gson().toJson(baseConfig)
        )

        val extraConfigJson = DiffusionLoadConfigResolver.buildExtraConfigJson(
            modelId = "local/${modelDir.absolutePath}",
            configPath = configPath
        )

        assertTrue(extraConfigJson.contains("\"diffusion_memory_mode\":\"1\""))
    }

    private fun createModelDir(name: String): File {
        val dir = File(application.filesDir, "diffusion-test/$name")
        dir.mkdirs()
        val baseConfig = ModelConfig.defaultConfig.deepCopy().apply {
            diffusionMemoryMode = "0"
        }
        File(dir, "config.json").writeText(
            Gson().toJson(baseConfig)
        )
        return dir
    }
}
