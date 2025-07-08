package com.alibaba.mnnllm.android.utils

import android.app.Application
import androidx.test.core.app.ApplicationProvider
import com.alibaba.mnnllm.android.model.ModelUtils
import io.mockk.every
import io.mockk.mockkStatic
import io.mockk.unmockkAll
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.io.File


@RunWith(RobolectricTestRunner::class)
@Config(sdk = [28])
class MmapUtilsTest {

    private lateinit var application: Application
    private lateinit var tempDir: File

    @Before
    fun setUp() {
        application = ApplicationProvider.getApplicationContext()
        tempDir = File(application.filesDir, "tmps")
        if (tempDir.exists()) {
            tempDir.deleteRecursively()
        }
        tempDir.mkdirs()

        mockkStatic(com.alibaba.mls.api.ApplicationProvider::class)
        every { com.alibaba.mls.api.ApplicationProvider.get() } returns application
    }

    @After
    fun tearDown() {
        if (tempDir.exists()) {
            tempDir.deleteRecursively()
        }
        unmockkAll()
    }

    @Test
    fun getMmapDir_withSimpleModelId() {
        val modelId = "test-model"
        val expected = "${application.filesDir.path}/tmps/test-model"
        val actual = MmapUtils.getMmapDir(modelId)
        assertEquals(expected, actual)
    }

    @Test
    fun getMmapDir_withModelScope() {
        val modelId = "ModelScope/test-model"
        val expected = "${application.filesDir.path}/tmps/test-model/modelscope"
        val actual = MmapUtils.getMmapDir(modelId)
        assertEquals(expected, actual)
    }

    @Test
    fun getMmapDir_withModelers() {
        val modelId = "Modelers/test-model"
        val expected = "${application.filesDir.path}/tmps/test-model/modelers"
        val actual = MmapUtils.getMmapDir(modelId)
        assertEquals(expected, actual)
    }

    @Test
    fun getMmapDir_withHuggingFace() {
        val modelId = "HuggingFace/test-model"
        val expected = "${application.filesDir.path}/tmps/test-model"
        val actual = MmapUtils.getMmapDir(modelId)
        assertEquals(expected, actual)
    }

    @Test
    fun getMmapDir_withMnnPrefix() {
        val modelId = "MNN/test-model"
        val expected = "${application.filesDir.path}/tmps/taobao-mnn_test-model"
        val actual = MmapUtils.getMmapDir(modelId)
        assertEquals(expected, actual)
    }


    @Test
    fun clearMmapCache() {
        val modelId = "test-model-to-delete"
        val mmapDir = MmapUtils.getMmapDir(modelId)
        val dir = File(mmapDir)
        dir.mkdirs()
        val testFile = File(dir, "test.txt")
        testFile.createNewFile()
        assertTrue(testFile.exists())
        assertTrue(dir.exists())

        val result = MmapUtils.clearMmapCache(modelId)

        assertTrue(result)
        assertFalse(dir.exists())
    }
} 