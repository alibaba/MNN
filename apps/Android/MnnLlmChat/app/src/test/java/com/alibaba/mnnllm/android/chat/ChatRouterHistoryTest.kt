package com.alibaba.mnnllm.android.chat

import android.app.Application
import androidx.test.core.app.ApplicationProvider
import com.alibaba.mls.api.download.ModelDownloadManager
import com.alibaba.mnnllm.android.model.ModelTypeUtils
import com.alibaba.mnnllm.android.model.ModelUtils
import com.alibaba.mnnllm.android.qnn.QnnModule
import io.mockk.*
import org.junit.After
import org.junit.Before
import org.junit.Test
import org.junit.Assert.assertEquals
import org.robolectric.Shadows.shadowOf
import java.io.File
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [28], application = Application::class)
class ChatRouterHistoryTest {
    private val downloads = mockk<ModelDownloadManager>(relaxed = true)
    private val modelId = "ModelScope/MNN/deleted-qnn"

    @Before
    fun setup() {
        mockkObject(ModelDownloadManager, ModelTypeUtils, ModelUtils, QnnModule)
        every { ModelDownloadManager.getInstance(any()) } returns downloads
        every { ModelTypeUtils.isDiffusionModel(any()) } returns false
        every { ModelTypeUtils.isQnnModel(any<String>()) } returns true
        every { ModelUtils.getConfigPathForModel(any<String>()) } returns null
    }

    @After
    fun teardown() = unmockkAll()

    @Test
    fun missingHistoryModelStopsBeforeQnnChecksOrAnyDownloadSideEffects() {
        ChatRouter.startRun(ApplicationProvider.getApplicationContext<Application>(), modelId, null, "saved", true)
        verify(exactly = 0) { QnnModule.deviceSupported() }
        verify(exactly = 0) { downloads.downloadQnnLibs() }
        verify(exactly = 0) { downloads.pauseAllDownloads() }
        verify(exactly = 0) { downloads.startDownload(any()) }
    }

    @Test
    fun staleConfigPathAlsoStopsBeforeQnnInitialization() {
        every { ModelUtils.getConfigPathForModel(modelId) } returns "/missing/config.json"
        ChatRouter.startRun(ApplicationProvider.getApplicationContext<Application>(), modelId, null, "saved", true)
        verify(exactly = 0) { QnnModule.deviceSupported() }
        verify(exactly = 0) { downloads.downloadQnnLibs() }
    }
    @Test
    fun directoryNamedConfigJsonIsNotAcceptedAsATextModelConfig() {
        val context = ApplicationProvider.getApplicationContext<Application>()
        val directory = File(context.cacheDir, "directory-config.json").apply { mkdirs() }
        try {
            every { ModelUtils.getConfigPathForModel(modelId) } returns directory.absolutePath
            ChatRouter.startRun(context, modelId, null, "saved", true)
            verify(exactly = 0) { QnnModule.deviceSupported() }
            verify(exactly = 0) { downloads.downloadQnnLibs() }
        } finally {
            directory.delete()
        }
    }

    @Test
    fun readableExistingConfigContinuesThroughNormalChatIntent() {
        val context = ApplicationProvider.getApplicationContext<Application>()
        val config = File(context.cacheDir, "history-config.json").apply { writeText("{}") }
        try {
            every { ModelUtils.getConfigPathForModel(modelId) } returns config.absolutePath
            every { ModelTypeUtils.isQnnModel(modelId) } returns false
            ChatRouter.startRun(context, modelId, null, "saved", true)
            val intent = shadowOf(context).nextStartedActivity
            assertEquals(ChatActivity::class.java.name, intent.component!!.className)
            assertEquals("saved", intent.getStringExtra("chatSessionId"))
            assertEquals(modelId, intent.getStringExtra("modelId"))
            assertEquals(config.absolutePath, intent.getStringExtra("configFilePath"))
        } finally {
            config.delete()
        }
    }

    @Test
    fun localDiffusionConfigFileContinuesWithItsParentDirectory() {
        val context = ApplicationProvider.getApplicationContext<Application>()
        val directory = File(context.cacheDir, "local-diffusion").apply { mkdirs() }
        val config = File(directory, "config.json").apply { writeText("{}") }
        val localModelId = "local/${directory.absolutePath}"
        try {
            every { ModelTypeUtils.isDiffusionModel(localModelId) } returns true
            every { ModelTypeUtils.isQnnModel(localModelId) } returns false
            every { ModelUtils.getConfigPathForModel(localModelId) } returns config.absolutePath
            ChatRouter.startRun(context, localModelId, null, "local-diffusion-history", true)
            val intent = shadowOf(context).nextStartedActivity
            assertEquals(directory.absolutePath, intent.getStringExtra("diffusionDir"))
            assertEquals(localModelId, intent.getStringExtra("modelId"))
            assertEquals("local-diffusion-history", intent.getStringExtra("chatSessionId"))
        } finally {
            directory.deleteRecursively()
        }
    }

    @Test
    fun downloadedDiffusionDirectoryRetainsTheExistingRouterContract() {
        val context = ApplicationProvider.getApplicationContext<Application>()
        val directory = File(context.cacheDir, "downloaded-diffusion").apply { mkdirs() }
        val diffusionModelId = "ModelScope/MNN/stable-diffusion"
        try {
            every { ModelTypeUtils.isDiffusionModel(diffusionModelId) } returns true
            every { ModelTypeUtils.isQnnModel(diffusionModelId) } returns false
            every { ModelUtils.getConfigPathForModel(diffusionModelId) } returns directory.absolutePath
            ChatRouter.startRun(context, diffusionModelId, null, "diffusion-history", true)
            val intent = shadowOf(context).nextStartedActivity
            assertEquals(directory.absolutePath, intent.getStringExtra("diffusionDir"))
            assertEquals(diffusionModelId, intent.getStringExtra("modelId"))
        } finally {
            directory.deleteRecursively()
        }
    }

}
