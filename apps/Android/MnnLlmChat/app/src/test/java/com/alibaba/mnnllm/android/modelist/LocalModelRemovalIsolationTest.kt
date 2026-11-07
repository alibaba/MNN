package com.alibaba.mnnllm.android.modelist

import android.app.Application
import android.content.Context
import androidx.test.core.app.ApplicationProvider
import com.alibaba.mls.api.download.DownloadListener
import com.alibaba.mls.api.download.DownloadPersistentData
import com.alibaba.mls.api.download.ModelDownloadManager
import io.mockk.mockk
import io.mockk.verify
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.io.File

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34], manifest = Config.NONE)
class LocalModelRemovalIsolationTest {
    private val application: Application get() = ApplicationProvider.getApplicationContext()
    private val previousApplication = com.alibaba.mls.api.ApplicationProvider.application

    @After
    fun cleanup() {
        File(application.filesDir, ".mnnmodels/modelscope/Example").delete()
        application.getSharedPreferences("DOWNLOAD_Example", Context.MODE_PRIVATE).edit().clear().commit()
        com.alibaba.mls.api.ApplicationProvider.set(previousApplication)
    }

    @Test
    fun `forgetting local import preserves same named remote files and legacy progress`() {
        com.alibaba.mls.api.ApplicationProvider.set(application)
        val remoteLink = File(application.filesDir, ".mnnmodels/modelscope/Example")
        remoteLink.parentFile!!.mkdirs()
        remoteLink.writeText("remote model link sentinel")
        val legacy = application.getSharedPreferences("DOWNLOAD_Example", Context.MODE_PRIVATE)
        legacy.edit().putLong(DownloadPersistentData.SIZE_SAVED_KEY, 123L).commit()
        val localId = "local/${File(application.filesDir, "imported_models/Example").absolutePath}"
        DownloadPersistentData.saveDownloadSizeSaved(application, localId, 456L)
        val manager = ModelDownloadManager(application)
        val listener = mockk<DownloadListener>(relaxed = true)
        manager.addListener(listener)

        manager.removeLocalModelState(localId)

        assertEquals(-1L, DownloadPersistentData.getDownloadSizeSaved(application, localId))
        assertTrue(remoteLink.isFile)
        assertEquals("remote model link sentinel", remoteLink.readText())
        assertEquals(123L, legacy.getLong(DownloadPersistentData.SIZE_SAVED_KEY, -1L))
        verify(exactly = 1) { listener.onDownloadFileRemoved(localId) }
    }

    @Test
    fun `local metadata removal refuses remote model ids without notifying listeners`() {
        com.alibaba.mls.api.ApplicationProvider.set(application)
        val manager = ModelDownloadManager(application)
        val listener = mockk<DownloadListener>(relaxed = true)
        manager.addListener(listener)
        try {
            manager.removeLocalModelState("ModelScope/MNN/Example")
            throw AssertionError("Expected remote model ID rejection")
        } catch (_: IllegalArgumentException) {
            verify(exactly = 0) { listener.onDownloadFileRemoved(any()) }
        }
    }
}
