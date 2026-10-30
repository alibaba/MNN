package com.alibaba.mnnllm.android.modelist

import android.app.Application
import androidx.test.core.app.ApplicationProvider
import com.alibaba.mls.api.download.ModelDownloadManager
import com.alibaba.mnnllm.android.chat.model.ChatDataManager
import com.alibaba.mnnllm.android.chat.model.SessionItem
import com.alibaba.mnnllm.android.history.HistoryUtils
import com.alibaba.mnnllm.android.utils.MmapUtils
import io.mockk.every
import io.mockk.mockk
import io.mockk.mockkObject
import io.mockk.unmockkAll
import io.mockk.verify
import org.junit.After
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [28], application = Application::class)
class RetainedHistoryDeletionTest {
    private lateinit var context: Application
    private lateinit var downloader: ModelDownloadManager
    private lateinit var chats: ChatDataManager
    private val modelId = "ModelScope/MNN/example"

    @Before
    fun setUp() {
        context = ApplicationProvider.getApplicationContext()
        downloader = mockk(relaxed = true)
        chats = mockk(relaxed = true)
        mockkObject(ModelDownloadManager, ChatDataManager, MmapUtils, HistoryUtils)
        every { ModelDownloadManager.getInstance(context) } returns downloader
        every { ChatDataManager.getInstance(context) } returns chats
        every { MmapUtils.clearMmapCache(modelId) } returns true
        every { chats.getSessionsForModel(modelId) } returns listOf(
            SessionItem("session-1", modelId, "First", 123L),
            SessionItem("session-2", modelId, "Second", 456L)
        )
        every { HistoryUtils.deleteHistory(any(), any(), any()) } returns Unit
    }

    @After
    fun tearDown() = unmockkAll()

    @Test
    fun keepHistoryDoesNotAccessOrDeleteChatStorage() {
        val result = ModelDeletionHelper.deleteModelWithCleanup(context, modelId, keepHistory = true)

        assertTrue(result.modelDeleted)
        assertTrue(result.mmapCacheCleared)
        assertFalse(result.chatSessionsCleared)
        assertTrue(result.errors.isEmpty())
        verify(exactly = 1) { MmapUtils.clearMmapCache(modelId) }
        verify(exactly = 1) { downloader.deleteModel(modelId) }
        verify(exactly = 0) { ChatDataManager.getInstance(any()) }
        verify(exactly = 0) { HistoryUtils.deleteHistory(any(), any(), any()) }
    }

    @Test
    fun defaultDeletionStillCleansOnlySelectedModelsSessions() {
        val result = ModelDeletionHelper.deleteModelWithCleanup(context, modelId)

        assertTrue(result.modelDeleted)
        assertTrue(result.chatSessionsCleared)
        verify(exactly = 1) { chats.getSessionsForModel(modelId) }
        verify(exactly = 1) { HistoryUtils.deleteHistory(context, chats, "session-1") }
        verify(exactly = 1) { HistoryUtils.deleteHistory(context, chats, "session-2") }
        verify(exactly = 1) { downloader.deleteModel(modelId) }
        verify(exactly = 0) { downloader.deleteModel("HuggingFace/MNN/example") }
    }

    @Test
    fun keepHistoryNeverFallsBackToChatDeletionWhenModelDeletionFails() {
        every { downloader.deleteModel(modelId) } throws IllegalStateException("disk failure")

        val result = ModelDeletionHelper.deleteModelWithCleanup(context, modelId, keepHistory = true)

        assertFalse(result.modelDeleted)
        assertFalse(result.chatSessionsCleared)
        assertTrue(result.errors.any { it.contains("disk failure") })
        verify(exactly = 0) { ChatDataManager.getInstance(any()) }
        verify(exactly = 0) { HistoryUtils.deleteHistory(any(), any(), any()) }
    }

    @Test
    fun keepHistoryStillReportsCacheCleanupFailure() {
        every { MmapUtils.clearMmapCache(modelId) } throws IllegalStateException("cache failure")

        val result = ModelDeletionHelper.deleteModelWithCleanup(context, modelId, keepHistory = true)

        assertTrue(result.modelDeleted)
        assertFalse(result.mmapCacheCleared)
        assertFalse(result.chatSessionsCleared)
        assertTrue(result.errors.any { it.contains("cache failure") })
        verify(exactly = 0) { HistoryUtils.deleteHistory(any(), any(), any()) }
    }
}
