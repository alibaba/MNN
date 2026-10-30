package com.alibaba.mnnllm.android.modelist

import android.app.Application
import android.content.Context
import android.net.Uri
import androidx.test.core.app.ApplicationProvider
import com.alibaba.mls.api.download.ModelDownloadManager
import com.alibaba.mnnllm.android.chat.chatlist.ChatViewHolders
import com.alibaba.mnnllm.android.chat.model.ChatDataItem
import com.alibaba.mnnllm.android.chat.model.ChatDataManager
import com.alibaba.mnnllm.android.utils.FileUtils
import com.alibaba.mnnllm.android.utils.MmapUtils
import io.mockk.every
import io.mockk.mockk
import io.mockk.mockkObject
import io.mockk.unmockkAll
import io.mockk.verify
import org.junit.After
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.io.File

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [28], application = Application::class)
class RetainedHistoryStorageTest {
    private lateinit var context: Application
    private lateinit var chats: ChatDataManager
    private lateinit var downloader: ModelDownloadManager
    private val selectedModel = "ModelScope/MNN/example"
    private val otherModel = "HuggingFace/MNN/example"
    private val sessions = listOf("retention-first", "retention-second", "retention-other")
    private val attachmentBytes = byteArrayOf(1, 3, 5, 7, 9)

    @Before
    fun setUp() {
        context = ApplicationProvider.getApplicationContext()
        context.deleteDatabase("chat.db")
        chats = newChatDataManager()
        downloader = mockk(relaxed = true)
        mockkObject(ChatDataManager, ModelDownloadManager, MmapUtils)
        every { ChatDataManager.getInstance(context) } returns chats
        every { ModelDownloadManager.getInstance(context) } returns downloader
        every { MmapUtils.clearMmapCache(selectedModel) } returns true
        seedSession(sessions[0], selectedModel)
        seedSession(sessions[1], selectedModel)
        seedSession(sessions[2], otherModel)
    }

    @After
    fun tearDown() {
        unmockkAll()
        sessions.forEach { resourceDirectory(it).deleteRecursively() }
        context.deleteDatabase("chat.db")
    }

    @Test
    fun retainedRowsAndAttachmentBytesSurviveDeletionAndDatabaseReopen() {
        val before = sessions.associateWith { snapshot(chats, it) }

        val result = ModelDeletionHelper.deleteModelWithCleanup(context, selectedModel, keepHistory = true)
        val reopened = newChatDataManager()

        assertTrue(result.modelDeleted)
        assertFalse(result.chatSessionsCleared)
        assertTrue(result.errors.isEmpty())
        sessions.forEach { sessionId ->
            assertEquals(before.getValue(sessionId), snapshot(reopened, sessionId))
            assertArrayEquals(attachmentBytes, File(resourceDirectory(sessionId), "image.png").readBytes())
        }
        verify(exactly = 1) { downloader.deleteModel(selectedModel) }
        verify(exactly = 0) { downloader.deleteModel(otherModel) }
    }

    @Test
    fun fullDeletionUsesRealHistoryCleanupOnlyForSelectedModel() {
        val otherBefore = snapshot(chats, sessions[2])

        val result = ModelDeletionHelper.deleteModelWithCleanup(context, selectedModel, keepHistory = false)
        val reopened = newChatDataManager()

        assertTrue(result.chatSessionsCleared)
        assertTrue(result.errors.isEmpty())
        assertTrue(reopened.getSessionsForModel(selectedModel).isEmpty())
        sessions.take(2).forEach { sessionId ->
            assertTrue(reopened.getChatDataBySession(sessionId).isEmpty())
            assertFalse(resourceDirectory(sessionId).exists())
        }
        assertEquals(otherBefore, snapshot(reopened, sessions[2]))
        assertArrayEquals(attachmentBytes, File(resourceDirectory(sessions[2]), "image.png").readBytes())
        verify(exactly = 1) { downloader.deleteModel(selectedModel) }
        verify(exactly = 0) { downloader.deleteModel(otherModel) }
    }

    private fun seedSession(sessionId: String, modelId: String) {
        val image = File(resourceDirectory(sessionId), "image.png")
        image.parentFile!!.mkdirs()
        image.writeBytes(attachmentBytes)
        chats.addOrUpdateSession(sessionId, modelId)
        chats.updateSessionName(sessionId, "Title $sessionId")
        chats.addChatData(sessionId, ChatDataItem("1000", ChatViewHolders.USER, "Question $sessionId").apply {
            imageUris = listOf(Uri.fromFile(image))
        })
        chats.addChatData(sessionId, ChatDataItem("2000", ChatViewHolders.ASSISTANT, "Answer $sessionId"))
        assertEquals(2, chats.getChatDataBySession(sessionId).size)
    }

    private fun snapshot(manager: ChatDataManager, sessionId: String): SessionSnapshot {
        val session = manager.allSessions.single { it.sessionId == sessionId }
        assertTrue(session.lastChatTime > 0)
        return SessionSnapshot(
            session.modelId, session.title, session.lastChatTime,
            manager.getChatDataBySession(sessionId).map {
                MessageSnapshot(it.time, it.type, it.text, it.displayText, it.imageUris?.map(Uri::toString))
            }
        )
    }

    private fun resourceDirectory(sessionId: String) = File(FileUtils.getSessionResourceBasePath(context, sessionId))

    // Isolate the real database client without replacing the process-wide singleton.
    private fun newChatDataManager(): ChatDataManager =
        ChatDataManager::class.java.getDeclaredConstructor(Context::class.java).apply {
            isAccessible = true
        }.newInstance(context)

    private data class SessionSnapshot(
        val modelId: String,
        val title: String,
        val lastChatTime: Long,
        val messages: List<MessageSnapshot>
    )

    private data class MessageSnapshot(
        val time: String?,
        val type: Int,
        val text: String?,
        val displayText: String?,
        val imageUris: List<String>?
    )
}
