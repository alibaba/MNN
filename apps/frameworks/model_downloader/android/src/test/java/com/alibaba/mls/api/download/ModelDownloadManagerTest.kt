package com.alibaba.mls.api.download

import android.content.Context
import org.robolectric.RuntimeEnvironment
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.io.File
import com.alibaba.mls.api.download.DownloadPersistentData
import com.alibaba.mls.api.ApplicationProvider

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33], manifest = Config.NONE)
class ModelDownloadManagerTest {

    @Test
    fun testProgressCallbackThrottle_ShouldLimitFrequentDownloadingCallbacks() {
        val context = RuntimeEnvironment.getApplication()
        ApplicationProvider.set(context)

        val manager = ModelDownloadManager(context)
        manager.setProgressCallbackIntervalMs(1000L)

        var progressCallbackCount = 0
        val listener = object : DownloadListener {
            override fun onDownloadStart(modelId: String) = Unit
            override fun onDownloadProgress(modelId: String, downloadInfo: DownloadInfo) {
                progressCallbackCount++
            }
            override fun onDownloadFinished(modelId: String, path: String) = Unit
            override fun onDownloadFailed(modelId: String, e: Exception) = Unit
        }
        manager.addListener(listener)

        val modelId = "ModelScope/MNN/ThrottleTestModel"
        manager.onDownloadingProgress(modelId, "file", "a.bin", 10L, 100L)
        manager.onDownloadingProgress(modelId, "file", "a.bin", 20L, 100L)
        manager.onDownloadingProgress(modelId, "file", "a.bin", 30L, 100L)

        assertEquals("Frequent DOWNLOADING callbacks should be throttled", 1, progressCallbackCount)
    }

    @Test
    fun testProgressCallbackThrottle_ShouldBypassThrottleWhenStageChanges() {
        val context = RuntimeEnvironment.getApplication()
        ApplicationProvider.set(context)

        val manager = ModelDownloadManager(context)
        manager.setProgressCallbackIntervalMs(1000L)

        var progressCallbackCount = 0
        val listener = object : DownloadListener {
            override fun onDownloadStart(modelId: String) = Unit
            override fun onDownloadProgress(modelId: String, downloadInfo: DownloadInfo) {
                progressCallbackCount++
            }
            override fun onDownloadFinished(modelId: String, path: String) = Unit
            override fun onDownloadFailed(modelId: String, e: Exception) = Unit
        }
        manager.addListener(listener)

        val modelId = "ModelScope/MNN/ThrottleStageTestModel"
        manager.onDownloadingProgress(modelId, "file", "a.bin", 10L, 100L)
        manager.onDownloadingProgress(modelId, "file", "a.bin", 20L, 100L)
        manager.onDownloadingProgress(modelId, "verify", "a.bin", 20L, 100L)

        assertEquals("Stage changes should force progress callback dispatch", 2, progressCallbackCount)
    }

    @Test
    fun testGetDownloadInfo_ReturnsNotStart_WhenPartialDownloadExists_ButNotInMemory() {
        val context = RuntimeEnvironment.getApplication()
        ApplicationProvider.set(context)
        
        val modelId = "Test/PartialModel"
        val totalSize = 1000L
        val savedSize = 500L

        // Pre-condition: Simulate a partial download in persistent storage
        DownloadPersistentData.saveDownloadSizeTotal(context, modelId, totalSize)
        DownloadPersistentData.saveDownloadSizeSaved(context, modelId, savedSize)
        
        // Also simulate the file existing (since getRealDownloadSize checks it)
        // We need to know where the file is expected to be.
        // ModelDownloadManager uses getDownloaderForSource -> getDownloadPath
        // This is a bit complex to simulate fully without understanding the path logic.
        // However, looking at MnnLlmChatOld's implementation, it checks PersistentData FIRST.;
        // Wait, MnnLlmChatOld:
        // getRealDownloadSize calls getDownloadSizeSaved first. If it returns < 0, then checks file.
        // So if we saveDownloadSizeSaved, getRealDownloadSize should return it without checking file.
        
        val manager = ModelDownloadManager(context)
        
        // Act: Get download info
        // Note: The first call might cache it, so we want to verify the initial fetch logic.
        // We assume the map is empty for this modelId initially.
        val info = manager.getDownloadInfo(modelId)
        
        // Assert: Currently, it likely returns NOT_START because it doesn't check persistent data for partials
        // We expect it to be PAUSED if it was loading correctly, but since it's buggy, we expect NOT_START or failure to detect pause.
        
        // Validate specifically what we are testing:
        // The issue is: "首次下载时候打开时候没有显示当前的下载进度，直到点击下载按钮才会显示下载进度"
        // This effectively means state is NOT_START instead of PAUSED (or showing progress).
        
        // In the buggy version, it returns NOT_START.
        // In the fixed version, it should return PAUSED (or at least have progress set).
        
        assertEquals("Should be PAUSED in fixed implementation", DownloadState.PAUSED, info.downloadState)
        assertEquals("Progress should be 0.5", 0.5, info.progress, 0.001)
    }

    @Test
    fun testDeleteModel_ClearsIncompleteProgressState() {
        val context = RuntimeEnvironment.getApplication()
        ApplicationProvider.set(context)

        val modelId = "Test/DeleteIncompleteModel"
        DownloadPersistentData.saveDownloadSizeTotal(context, modelId, 1000L)
        DownloadPersistentData.saveDownloadSizeSaved(context, modelId, 200L)

        val manager = ModelDownloadManager(context)
        val beforeDelete = manager.getDownloadInfo(modelId)
        assertEquals(DownloadState.PAUSED, beforeDelete.downloadState)

        manager.deleteModel(modelId)

        val afterDelete = manager.getDownloadInfo(modelId)
        assertEquals("deleteModel should reset incomplete model to NOT_START", DownloadState.NOT_START, afterDelete.downloadState)
        assertEquals("deleteModel should clear saved size", 0L, afterDelete.savedSize)
        assertEquals("deleteModel should clear progress", 0.0, afterDelete.progress, 0.0)
        assertEquals("deleteModel should keep total size for pre-download UI display", 1000L, afterDelete.totalSize)
    }

    @Test
    fun testGetDownloadedFile_shouldNotReturnOtherSourcePath() {
        val context = RuntimeEnvironment.getApplication()
        ApplicationProvider.set(context)

        val hfModelId = "HuggingFace/taobao-mnn/MiniMind2-MNN"
        val msModelId = "ModelScope/MNN/MiniMind2-MNN"

        val hfLink = File(context.filesDir, ".mnnmodels/huggingface/MiniMind2-MNN")
        hfLink.parentFile?.mkdirs()
        hfLink.mkdirs()

        val manager = ModelDownloadManager(context)
        val msDownloadedFile = manager.getDownloadedFile(msModelId)

        assertNull(
            "ModelScope query should not return HuggingFace link path with same tail name",
            msDownloadedFile
        )

        // cleanup
        hfLink.deleteRecursively()
    }

    @Test
    fun testDeleteModel_shouldRemoveHfStorageFolderUsingFullModelIdNaming() {
        val context = RuntimeEnvironment.getApplication()
        ApplicationProvider.set(context)

        val modelId = "HuggingFace/taobao-mnn/MiniMind2-MNN"
        val hfStorage = File(
            context.filesDir,
            ".mnnmodels/models--HuggingFace--taobao-mnn--MiniMind2-MNN/blobs"
        )
        val hfLink = File(context.filesDir, ".mnnmodels/MiniMind2-MNN")
        hfStorage.mkdirs()
        hfLink.mkdirs()

        val manager = ModelDownloadManager(context)
        manager.deleteModel(modelId)

        assertTrue("HF storage folder should be removed by deleteModel", !hfStorage.parentFile!!.exists())
        assertTrue("HF link folder should be removed by deleteModel", !hfLink.exists())
    }

    @Test
    fun testPausedDownload_shouldRestorePausedStateAfterManagerRecreate() {
        val context = RuntimeEnvironment.getApplication()
        ApplicationProvider.set(context)

        val modelId = "ModelScope/MNN/ResumeAfterRestart"
        val manager = ModelDownloadManager(context)

        // Simulate an in-progress download then pause.
        manager.onDownloadingProgress(modelId, "file", "a.bin", 200L, 1000L)
        manager.onDownloadPaused(modelId)

        // Simulate app restart by recreating manager (in-memory cache cleared).
        val recreated = ModelDownloadManager(context)
        val info = recreated.getDownloadInfo(modelId)

        assertEquals("Paused download should be restored after restart", DownloadState.PAUSED, info.downloadState)
        assertEquals("Saved size should be restored after restart", 200L, info.savedSize)
        assertEquals("Total size should be restored after restart", 1000L, info.totalSize)
        assertEquals("Progress should be restored after restart", 0.2, info.progress, 0.001)
    }

    @Test
    fun testOnRepoInfo_shouldNotMarkUpdate_whenRemoteOlderThanLocalDownloadTime() {
        val context = RuntimeEnvironment.getApplication()
        ApplicationProvider.set(context)

        val modelId = "ModelScope/MNN/Fun-Audio-Chat-8B-MNN"
        val manager = ModelDownloadManager(context)
        manager.onDownloadFileFinished(modelId, "/tmp/fake")

        val localDownloadedTime = 1_777_513_918_172L // 2026-03-03
        val remoteLastModified = 1_770_000_000_000L // older than local
        DownloadPersistentData.saveDownloadedTime(context, modelId, localDownloadedTime)

        manager.onRepoInfo(modelId, remoteLastModified, 6_400_000_000L)

        val info = manager.getDownloadInfo(modelId)
        assertFalse("Remote older than local download time should not be marked as update", info.hasUpdate)
    }

    @Test
    fun testOnRepoInfo_shouldMarkUpdate_whenRemoteNewerThanLocalDownloadTime() {
        val context = RuntimeEnvironment.getApplication()
        ApplicationProvider.set(context)

        val modelId = "ModelScope/MNN/Qwen3.5-0.8B-MNN"
        val manager = ModelDownloadManager(context)
        manager.onDownloadFileFinished(modelId, "/tmp/fake")

        val localDownloadedTime = 1_770_000_000_000L
        val remoteLastModified = 1_777_513_918_172L // newer than local
        DownloadPersistentData.saveDownloadedTime(context, modelId, localDownloadedTime)

        manager.onRepoInfo(modelId, remoteLastModified, 547_000_000L)

        val info = manager.getDownloadInfo(modelId)
        assertTrue("Remote newer than local download time should be marked as update", info.hasUpdate)
    }

    @Test
    fun testOnRepoInfo_shouldNotMarkUpdate_whenRemoteTimeUnknownAndSizeUnchanged() {
        val context = RuntimeEnvironment.getApplication()
        ApplicationProvider.set(context)

        val modelId = "ModelScope/MNN/UnknownTimeNoChange"
        val manager = ModelDownloadManager(context)
        manager.onDownloadFileFinished(modelId, "/tmp/fake")

        val localTotalSize = 1_000_000L
        DownloadPersistentData.saveDownloadSizeTotal(context, modelId, localTotalSize)
        DownloadPersistentData.saveDownloadedTime(context, modelId, 0L)

        manager.onRepoInfo(modelId, 0L, localTotalSize)

        val info = manager.getDownloadInfo(modelId)
        assertFalse("Unknown remote time with unchanged size should not be marked as update", info.hasUpdate)
    }

    @Test
    fun testOnRepoInfo_shouldMarkUpdate_whenRemoteTimeUnknownAndSizeChanged() {
        val context = RuntimeEnvironment.getApplication()
        ApplicationProvider.set(context)

        val modelId = "ModelScope/MNN/UnknownTimeSizeChanged"
        val manager = ModelDownloadManager(context)
        manager.onDownloadFileFinished(modelId, "/tmp/fake")

        DownloadPersistentData.saveDownloadSizeTotal(context, modelId, 1_000_000L)
        DownloadPersistentData.saveDownloadedTime(context, modelId, 0L)

        manager.onRepoInfo(modelId, 0L, 1_100_000L)

        val info = manager.getDownloadInfo(modelId)
        assertTrue("Unknown remote time with changed size should be marked as update", info.hasUpdate)
    }
}
