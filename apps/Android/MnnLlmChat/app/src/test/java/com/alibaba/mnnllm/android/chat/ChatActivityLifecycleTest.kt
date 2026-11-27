package com.alibaba.mnnllm.android.chat

import android.content.Intent
import androidx.test.core.app.ApplicationProvider
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleRegistry
import com.alibaba.mnnllm.android.chat.model.ChatDataItem
import com.alibaba.mnnllm.android.chat.model.ChatDataManager
import com.alibaba.mnnllm.android.llm.ChatService
import com.alibaba.mnnllm.android.llm.ChatSession
import com.alibaba.mnnllm.android.llm.GenerateProgressListener
import io.mockk.every
import io.mockk.mockk
import io.mockk.mockkObject
import io.mockk.spyk
import io.mockk.just
import io.mockk.Runs
import io.mockk.unmockkAll
import io.mockk.verify
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.test.UnconfinedTestDispatcher
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Assert.*
import java.util.concurrent.CountDownLatch
import java.util.concurrent.ExecutionException
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.Robolectric
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [28])
@OptIn(kotlinx.coroutines.ExperimentalCoroutinesApi::class)
class ChatActivityLifecycleTest {
    @After
    fun tearDown() {
        unmockkAll()
        Dispatchers.resetMain()
    }

    private fun presenter(session: ChatSession): Pair<ChatPresenter, ChatActivity> {
        Dispatchers.setMain(UnconfinedTestDispatcher())
        val activity = mockk<ChatActivity>(relaxed = true)
        val lifecycle = LifecycleRegistry.createUnsafe(activity)
        every { activity.lifecycle } returns lifecycle
        lifecycle.currentState = Lifecycle.State.RESUMED
        mockkObject(ChatDataManager.Companion)
        every { ChatDataManager.getInstance(any()) } returns mockk(relaxed = true)
        val presenter = ChatPresenter(activity, "test-model", "test-model")
        ChatPresenter::class.java.getDeclaredField("chatSession").apply { isAccessible = true }.set(presenter, session)
        ChatPresenter::class.java.getDeclaredField("sessionId").apply { isAccessible = true }.set(presenter, "test-session")
        return presenter to activity
    }

    @Test
    fun loadFailure_doesNotDispatchSuccessfulLoadingCallback() {
        val session = mockk<ChatSession>(relaxed = true)
        every { session.load() } throws IllegalStateException("injected load failure")
        val (presenter, activity) = presenter(session)
        presenter.load()
        verify(timeout = 3000) { activity.onModelLoadFailed("injected load failure") }
        verify(timeout = 3000) { session.release() }
        verify(exactly = 0) { activity.onLoadingChanged(false, true) }
        presenter.destroy()
    }

    @Test
    fun loadOOM_reachesExistingErrorUi() {
        val session = mockk<ChatSession>(relaxed = true)
        every { session.load() } throws OutOfMemoryError("injected load OOM")
        val (presenter, activity) = presenter(session)
        presenter.load()
        verify(timeout = 3000) { activity.onModelLoadFailed("Not enough memory to load the model") }
        verify(timeout = 3000) { session.release() }
        verify(exactly = 0) { activity.onLoadingChanged(false, true) }
        presenter.destroy()
    }

    @Test
    fun generateOOM_preservesErrorResult_andNotifiesListener() = runBlocking {
        val session = mockk<ChatSession>(relaxed = true)
        every { session.generate(any(), any(), any()) } throws OutOfMemoryError("injected generation OOM")
        val (presenter, _) = presenter(session)
        val listener = mockk<ChatPresenter.GenerateListener>(relaxed = true)
        val result = presenter.requestGenerate(ChatDataItem(2).apply { text = "hello" }, listener)
        assertEquals(true, result["error"])
        assertEquals(true, result["session_invalid"])
        assertTrue((result["message"] as String).contains("Not enough memory"))
        verify { session.release() }
        verify(exactly = 1) { listener.onGenerateFinished(result) }
        presenter.destroy()
    }

    @Test
    fun cancelledGeneration_retainsOwnershipUntilNativeReturns() = runBlocking {
        val entered = CountDownLatch(1)
        val finish = CountDownLatch(1)
        val session = mockk<ChatSession>(relaxed = true)
        every { session.generate(any(), any(), any()) } answers {
            entered.countDown()
            assertTrue(finish.await(3, TimeUnit.SECONDS))
            assertTrue(thirdArg<GenerateProgressListener>().onProgress("done"))
            hashMapOf("response" to "done")
        }
        val (presenter, _) = presenter(session)
        val firstListener = mockk<ChatPresenter.GenerateListener>(relaxed = true)
        val secondListener = mockk<ChatPresenter.GenerateListener>(relaxed = true)
        val first = launch(Dispatchers.IO) {
            presenter.requestGenerate(ChatDataItem(2).apply { text = "first" }, firstListener)
        }
        try {
            assertTrue(entered.await(2, TimeUnit.SECONDS))
            first.cancel()
            val second = presenter.requestGenerate(ChatDataItem(2).apply { text = "second" }, secondListener)
            assertEquals("Generation already in progress", second["message"])
            verify(exactly = 0) { secondListener.onGenerateStart() }
            verify(exactly = 0) { session.release() }
            finish.countDown()
            first.join()
            assertTrue(first.isCancelled)
            verify(exactly = 1) { session.generate(any(), any(), any()) }
        } finally {
            finish.countDown()
            first.cancelAndJoin()
            presenter.destroy()
        }
    }

    @Test
    fun modelReplacement_waitsForNativeRelease_andIgnoresStaleCompletion() {
        val service = ChatService()
        val previous = mockk<ChatSession>(relaxed = true)
        val next = mockk<ChatSession>(relaxed = true)
        val stale = mockk<ChatSession>(relaxed = true)
        val releaseStarted = CountDownLatch(1)
        every { previous.release() } answers { releaseStarted.countDown() }
        service.beforeLoad(previous)
        val executor = Executors.newSingleThreadExecutor()
        val replacement = executor.submit<Unit> { service.beforeLoad(next) }
        try {
            assertTrue(releaseStarted.await(3, TimeUnit.SECONDS))
            service.onNativeReleased(stale, succeeded = true)
            assertFalse(replacement.isDone)
            service.onNativeReleased(previous, succeeded = true)
            replacement.get(3, TimeUnit.SECONDS)
            verify(exactly = 1) { previous.release() }
            verify(exactly = 0) { next.release() }
        } finally {
            service.onNativeReleased(previous, succeeded = true)
            executor.shutdownNow()
        }
    }

    @Test
    fun releasedLoadWaiter_doesNotReleaseNewResident() {
        val service = ChatService()
        val previous = mockk<ChatSession>(relaxed = true)
        val stale = mockk<ChatSession>(relaxed = true)
        val next = mockk<ChatSession>(relaxed = true)
        val releaseRequested = AtomicBoolean(false)
        service.beforeLoad(previous)
        every { previous.release() } answers {
            releaseRequested.set(true)
            service.onNativeReleased(previous, succeeded = true)
            service.beforeLoad(next)
        }
        every { next.release() } answers { service.onNativeReleased(next, succeeded = true) }
        val executor = Executors.newSingleThreadExecutor()
        val replacement = executor.submit<Unit> { service.beforeLoad(stale) { releaseRequested.get() } }
        try {
            val error = assertThrows(ExecutionException::class.java) { replacement.get(3, TimeUnit.SECONDS) }
            assertTrue(error.cause is CancellationException)
            service.onNativeReleased(stale, succeeded = false)
            service.onNativeReleased(stale, succeeded = true)
            verify(exactly = 0) { next.release() }
            service.beforeLoad(next)
        } finally {
            executor.shutdownNow()
        }
    }

    @Test
    fun failedNativeRelease_doesNotAdmitReplacement() {
        val service = ChatService()
        val previous = mockk<ChatSession>(relaxed = true)
        val next = mockk<ChatSession>(relaxed = true)
        service.beforeLoad(previous)
        every { previous.release() } answers { service.onNativeReleased(previous, succeeded = false) }
        val executor = Executors.newSingleThreadExecutor()
        val replacement = executor.submit<Unit> { service.beforeLoad(next) }
        try {
            val error = assertThrows(ExecutionException::class.java) { replacement.get(3, TimeUnit.SECONDS) }
            assertEquals("Previous model could not be released", error.cause?.message)
            service.onNativeReleased(next, succeeded = true)
            verify(exactly = 0) { next.release() }
        } finally {
            executor.shutdownNow()
        }
    }

    @Test
    fun synchronousCreateFailure_reachesErrorUiWithoutLoadingModel() {
        Dispatchers.setMain(UnconfinedTestDispatcher())
        val activity = spyk(ChatActivity())
        val lifecycle = LifecycleRegistry.createUnsafe(activity)
        every { activity.lifecycle } returns lifecycle
        lifecycle.currentState = Lifecycle.State.RESUMED
        every { activity.onLoadingChanged(any(), any()) } just Runs
        every { activity.onModelLoadFailed(any()) } just Runs
        val presenter = mockk<ChatPresenter>(relaxed = true)
        every { presenter.createSession() } throws IllegalStateException("synchronous create failure")
        ChatActivity::class.java.getDeclaredField("chatPresenter").apply { isAccessible = true }.set(activity, presenter)
        ChatActivity::class.java.getDeclaredMethod("setupSession").apply { isAccessible = true }.invoke(activity)
        verify(timeout = 3000) { activity.onModelLoadFailed("synchronous create failure") }
        verify { presenter.destroy() }
        verify(exactly = 0) { presenter.load() }
        lifecycle.currentState = Lifecycle.State.DESTROYED
    }

    @Test
    fun stopRequest_keepsGeneratingStateUntilCallFinishes() {
        val activity = ChatActivity()
        val presenter = mockk<ChatPresenter>(relaxed = true)
        ChatActivity::class.java.getDeclaredField("chatPresenter").apply { isAccessible = true }.set(activity, presenter)
        activity.isGenerating = true
        activity.onStopGenerationRequested()
        assertTrue(activity.isGenerating)
        verify(exactly = 1) { presenter.stopGenerate() }
    }

    @Test
    fun destroy_withoutModelExtras_doesNotCrash() {
        val intent = Intent(ApplicationProvider.getApplicationContext(), ChatActivity::class.java)

        Robolectric.buildActivity(ChatActivity::class.java, intent)
            .create()
            .destroy()
    }
}
