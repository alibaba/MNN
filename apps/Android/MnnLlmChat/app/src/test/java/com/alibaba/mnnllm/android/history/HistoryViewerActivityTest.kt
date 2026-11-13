package com.alibaba.mnnllm.android.history

import android.app.Application
import android.content.Intent
import android.os.Looper
import android.widget.TextView
import android.widget.Button
import androidx.recyclerview.widget.RecyclerView
import androidx.test.core.app.ApplicationProvider
import com.alibaba.mls.api.download.ModelDownloadManager
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.chat.ChatRouter
import com.alibaba.mnnllm.android.chat.chatlist.ChatViewHolders
import com.alibaba.mnnllm.android.chat.model.ChatDataItem
import com.alibaba.mnnllm.android.chat.model.ChatDataManager
import com.alibaba.mnnllm.android.chat.model.SessionItem
import io.mockk.*
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.Robolectric
import org.robolectric.RobolectricTestRunner
import org.robolectric.Shadows.shadowOf
import org.robolectric.annotation.Config
import org.robolectric.android.controller.ActivityController
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [28], application = Application::class)
class HistoryViewerActivityTest {
    private val manager = mockk<ChatDataManager>(relaxed = true)
    private var controller: ActivityController<HistoryViewerActivity>? = null
    private val stored = SessionItem("session-1", "ModelScope/MNN/deleted-qnn", "Saved chat", 123L)

    @Before
    fun setup() {
        mockkObject(ChatDataManager, ChatRouter, ModelDownloadManager)
        every { ChatDataManager.getInstance(any()) } returns manager
        every { manager.allSessions } returns mutableListOf(stored)
        every { manager.getChatDataBySession(stored.sessionId) } returns listOf(
            ChatDataItem("123", ChatViewHolders.USER, "question"),
            ChatDataItem("124", ChatViewHolders.ASSISTANT, "answer")
        )
        every { ChatRouter.startRun(any(), any(), any(), any(), any()) } just Runs
    }

    @After
    fun tearDown() {
        controller?.pause()?.stop()?.destroy()
        unmockkAll()
    }

    private fun open(sessionId: String = stored.sessionId): HistoryViewerActivity {
        val context = ApplicationProvider.getApplicationContext<Application>()
        val intent = Intent(context, HistoryViewerActivity::class.java)
            .putExtra(HistoryViewerActivity.EXTRA_SESSION_ID, sessionId)
        controller = Robolectric.buildActivity(HistoryViewerActivity::class.java, intent)
        controller!!.get().setTheme(R.style.AppTheme)
        return controller!!.create().start().resume().get()
    }

    private fun awaitLoaded(activity: HistoryViewerActivity) {
        val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5)
        while (!activity.findViewById<Button>(R.id.history_continue).isEnabled && System.nanoTime() < deadline) {
            shadowOf(Looper.getMainLooper()).idle()
            Thread.sleep(10)
        }
        assertTrue(activity.findViewById<Button>(R.id.history_continue).isEnabled)
    }

    @Test
    fun viewingAndReopeningHistoryNeverStartsAnEngineOrDownloadManager() {
        val activity = open()
        awaitLoaded(activity)
        assertEquals(2, activity.findViewById<RecyclerView>(R.id.history_messages).adapter!!.itemCount)
        controller!!.pause().resume()
        awaitLoaded(activity)
        assertEquals(2, activity.findViewById<RecyclerView>(R.id.history_messages).adapter!!.itemCount)
        verify(exactly = 0) { ChatRouter.startRun(any(), any(), any(), any(), any()) }
        verify(exactly = 0) { ModelDownloadManager.getInstance(any()) }
        verify(exactly = 0) { manager.addOrUpdateSession(any(), any()) }
        verify(exactly = 0) { manager.addChatData(any(), any()) }
        verify(exactly = 0) { manager.updateSessionModelId(any(), any()) }
    }

    @Test
    fun onlyExplicitContinueUsesTheStoredModelIdentityAndRequiresExistingFiles() {
        val activity = open()
        awaitLoaded(activity)
        activity.findViewById<Button>(R.id.history_continue).performClick()
        verify(exactly = 1) {
            ChatRouter.startRun(activity, stored.modelId, null, stored.sessionId, requireExistingModel = true)
        }
    }

    @Test
    fun absentSessionIdDoesNotCreateOrResumeAConversation() {
        val activity = open("")
        assertFalse(activity.findViewById<Button>(R.id.history_continue).isEnabled)
        assertEquals(0, activity.findViewById<RecyclerView>(R.id.history_messages).adapter!!.itemCount)
        verify(exactly = 0) { ChatRouter.startRun(any(), any(), any(), any(), any()) }
        verify(exactly = 0) { ModelDownloadManager.getInstance(any()) }
        verify(exactly = 0) { manager.addOrUpdateSession(any(), any()) }
    }
    @Test
    fun deletedSessionShowsUnavailableWithoutRecreatingRows() {
        every { manager.allSessions } returns mutableListOf()
        val activity = open()
        val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5)
        val status = activity.findViewById<TextView>(R.id.history_status)
        val expected = activity.getString(R.string.history_session_unavailable)
        while (status.text.toString() != expected && System.nanoTime() < deadline) {
            shadowOf(Looper.getMainLooper()).idle()
            Thread.sleep(10)
        }
        assertEquals(expected, status.text.toString())
        assertFalse(activity.findViewById<Button>(R.id.history_continue).isEnabled)
        verify(exactly = 0) { manager.getChatDataBySession(any()) }
        verify(exactly = 0) { manager.addOrUpdateSession(any(), any()) }
        verify(exactly = 0) { ModelDownloadManager.getInstance(any()) }
    }

    @Test
    fun destructionWhileLoadingDoesNotResumeChatOrApplyStaleUiWork() {
        val started = CountDownLatch(1)
        val release = CountDownLatch(1)
        val completed = CountDownLatch(1)
        every { manager.allSessions } answers {
            started.countDown()
            release.await(5, TimeUnit.SECONDS)
            mutableListOf(stored)
        }
        every { manager.getChatDataBySession(stored.sessionId) } answers {
            completed.countDown()
            emptyList()
        }
        try {
            val activity = open()
            shadowOf(Looper.getMainLooper()).idle()
            assertTrue(started.await(5, TimeUnit.SECONDS))
            controller!!.pause().stop().destroy()
            controller = null
            release.countDown()
            assertTrue(completed.await(5, TimeUnit.SECONDS))
            shadowOf(Looper.getMainLooper()).idle()
            assertFalse(activity.findViewById<Button>(R.id.history_continue).isEnabled)
            verify(exactly = 0) { ChatRouter.startRun(any(), any(), any(), any(), any()) }
            verify(exactly = 0) { ModelDownloadManager.getInstance(any()) }
        } finally {
            release.countDown()
        }
    }

}
