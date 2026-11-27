package com.alibaba.mnnllm.android.chat

import android.app.Application
import android.net.Uri
import android.content.Intent
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import android.os.Looper
import android.view.View
import androidx.appcompat.view.menu.MenuBuilder
import com.alibaba.mnnllm.android.chat.agent.AgentStoredStep
import com.alibaba.mnnllm.android.chat.agent.AgentToolStepRecord
import com.alibaba.mnnllm.android.chat.agent.AgentStep
import com.alibaba.mnnllm.android.chat.agent.AgentStepKind
import com.alibaba.mnnllm.android.llm.ChatService
import com.alibaba.mnnllm.android.llm.GenerateProgressListener
import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.spyk
import java.util.concurrent.atomic.AtomicBoolean
import android.widget.TextView
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleRegistry
import androidx.test.core.app.ApplicationProvider
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.chat.agent.AgentBlockingInference
import com.alibaba.mnnllm.android.chat.agent.AgentConsentController
import com.alibaba.mnnllm.android.chat.agent.AgentGeneration
import com.alibaba.mnnllm.android.chat.agent.AgentInference
import com.alibaba.mnnllm.android.chat.agent.AgentInferenceRequest
import com.alibaba.mnnllm.android.chat.agent.AgentMessage
import com.alibaba.mnnllm.android.chat.agent.AgentPhase
import com.alibaba.mnnllm.android.chat.agent.AgentRequest
import com.alibaba.mnnllm.android.chat.agent.AgentRole
import com.alibaba.mnnllm.android.chat.agent.AgentRunState
import com.alibaba.mnnllm.android.chat.agent.AgentSkill
import com.alibaba.mnnllm.android.chat.agent.AgentStepStatus
import com.alibaba.mnnllm.android.chat.agent.AgentToolExecutor
import com.alibaba.mnnllm.android.chat.agent.AgentToolResult
import com.alibaba.mnnllm.android.chat.agent.SqliteAgentStore
import com.alibaba.mnnllm.android.chat.chatlist.ChatListComponent
import com.alibaba.mnnllm.android.chat.chatlist.ChatViewHolders
import com.alibaba.mnnllm.android.chat.input.ChatInputComponent
import com.alibaba.mnnllm.android.chat.model.ChatDataItem
import com.alibaba.mnnllm.android.chat.model.ChatDataManager
import com.alibaba.mnnllm.android.llm.ChatSession
import com.alibaba.mnnllm.android.llm.LlmSession
import com.alibaba.mnnllm.api.openai.di.ServiceLocator
import com.alibaba.mnnllm.api.openai.manager.ServerEventManager
import com.alibaba.mnnllm.api.openai.runtime.DefaultLlmRuntimeController
import kotlinx.coroutines.Job
import org.robolectric.annotation.Implementation
import org.robolectric.annotation.Implements
import io.mockk.every
import io.mockk.mockk
import io.mockk.mockkObject
import io.mockk.slot
import io.mockk.unmockkAll
import io.mockk.verify
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.awaitCancellation
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.UnconfinedTestDispatcher
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.Robolectric
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.util.UUID
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

@RunWith(RobolectricTestRunner::class)
@Config(application = Application::class, sdk = [28])
@OptIn(ExperimentalCoroutinesApi::class)
class AgentChatIntegrationTest {
    private val stores = mutableListOf<SqliteAgentStore>()
    private val activities = mutableListOf<org.robolectric.android.controller.ActivityController<ChatActivity>>()

    @After
    fun tearDown() {
        activities.forEach { it.destroy() }
        runBlocking { stores.forEach { it.close() } }
        unmockkAll()
        Dispatchers.resetMain()
    }

    private fun integration(dispatcher: kotlinx.coroutines.CoroutineDispatcher): AgentChatIntegration {
        val consent = AgentConsentController(dispatcher)
        val store = SqliteAgentStore(ApplicationProvider.getApplicationContext(), consent,
            databaseName = "agent-integration-${UUID.randomUUID()}.sqlite3", dispatcher = Dispatchers.IO)
        stores.add(store)
        return AgentChatIntegration({ store }, consent, dispatcher)
    }

    private class FakeInference(private vararg val outputs: String) : AgentInference {
        val requests = mutableListOf<AgentInferenceRequest>()
        override suspend fun countTokens(messages: List<AgentMessage>) = 100
        override suspend fun generate(request: AgentInferenceRequest, onDelta: (String) -> Unit): AgentGeneration {
            val text = outputs[requests.size]
            requests.add(request)
            text.chunked(7).forEach(onDelta)
            return AgentGeneration(text)
        }
    }

    private class FakeSession(
        override val sessionId: String,
        private val history: List<ChatDataItem> = emptyList(),
        private val loader: (FakeSession) -> Unit = {},
        private val releaser: (FakeSession) -> Unit = {},
    ) : ChatSession {
        val releaseRequested = AtomicBoolean(false)
        override val debugInfo = "fake lifecycle only"
        override val supportOmni = false
        override fun load() { loader(this) }
        override fun release() { if (releaseRequested.compareAndSet(false, true)) releaser(this) }
        override fun getHistory() = history
        override fun setHistory(history: List<ChatDataItem>?) { throw AssertionError("History must not be overwritten") }
        override fun reset() = sessionId
        override fun setKeepHistory(keepHistory: Boolean) { }
        override fun setEnableAudioOutput(enable: Boolean) { }
        override fun updateThinking(thinking: Boolean) { }
        override fun generate(prompt: String, params: Map<String, Any>, progressListener: GenerateProgressListener): HashMap<String, Any> {
            throw AssertionError("No real or fake generation expected in this lifecycle test")
        }
    }

    private fun presenterActivity(): ChatActivity {
        val activity = mockk<ChatActivity>(relaxed = true)
        every { activity.intent } returns Intent()
        every { activity.modelId } returns "Qwen3.5-0.8B"
        mockkObject(ChatDataManager.Companion)
        every { ChatDataManager.getInstance(any()) } returns mockk(relaxed = true)
        return activity
    }

    private val noTools = AgentToolExecutor { _, _ -> throw AssertionError("Unexpected tool execution") }
    private fun user(text: String) = ChatDataItem(ChatViewHolders.USER).apply { this.text = text }
    private fun assistant(text: String) = ChatDataItem(ChatViewHolders.ASSISTANT).apply { this.text = text }

    private suspend fun rejected(block: suspend () -> Unit) {
        try { block(); fail("Expected rejection") } catch (_: IllegalStateException) { }
    }

    @Test
    fun defaultOff_neverInfersOrSavesWithoutConfirmation() = runTest {
        val chat = integration(StandardTestDispatcher(testScheduler))
        chat.restore(null, emptyList())
        val inference = FakeInference("must not run")
        assertFalse(chat.mode.value.agent)
        assertFalse(chat.consent.state.value.enabled)
        rejected { chat.run(inference, AgentRequest("hello"), noTools) }
        assertTrue(inference.requests.isEmpty())
        assertTrue(chat.store.allMemories().isEmpty())
        assertTrue(chat.store.allSkills().isEmpty())
    }

    @Test
    fun modeChangesOnlyBeforeFirstMessage_andReopenRequiresFreshConsent() = runTest {
        val chat = integration(StandardTestDispatcher(testScheduler))
        chat.changeMode("history", true)
        assertTrue(chat.store.isAgentSession("history"))
        assertTrue(chat.mode.value.authorized)
        chat.changeMode("history", false)
        assertFalse(chat.store.isAgentSession("history"))
        assertFalse(chat.consent.state.value.networkAllowed)
        chat.changeMode("history", true)
        chat.restore("history", listOf(user("earlier")))
        assertTrue(chat.mode.value.agent)
        assertFalse(chat.mode.value.authorized)
        assertFalse(chat.mode.value.canChange)
        val inference = FakeInference("final")
        rejected { chat.run(inference, AgentRequest("hello", sessionId = "history"), noTools) }
        rejected { chat.changeMode("history", false) }
        chat.confirm()
        assertEquals(AgentPhase.COMPLETED, chat.run(inference,
            AgentRequest("hello", sessionId = "history"), noTools).phase)
    }

    @Test
    fun ordinaryConversationCannotBeConvertedAfterSending() = runTest {
        val chat = integration(StandardTestDispatcher(testScheduler))
        chat.restore("ordinary", listOf(user("hello"), assistant("hi")))
        rejected { chat.changeMode("ordinary", true) }
        assertFalse(chat.mode.value.agent)
        assertFalse(chat.consent.state.value.enabled)
    }

    @Test
    fun fullHistoryHasOneCurrentUser_andNeverReusesPreviousToolObservations() = runTest {
        val chat = integration(StandardTestDispatcher(testScheduler))
        chat.changeMode("history", true)
        val history = listOf(ChatDataItem(ChatViewHolders.HEADER), user("older user"),
            assistant("<think>hidden thought</think><answer>older answer</answer>"))
        val inference = FakeInference("<answer>new answer</answer>")
        chat.run(inference, AgentRequest("UNIQUE_CURRENT_USER",
            history = AgentChatIntegration.history(history), sessionId = "history"), noTools)
        val messages = inference.requests.single().messages
        assertEquals(listOf(AgentRole.SYSTEM, AgentRole.USER, AgentRole.ASSISTANT, AgentRole.USER), messages.map { it.role })
        assertEquals("older answer", messages[2].content)
        assertEquals(1, messages.count { it.content.contains("UNIQUE_CURRENT_USER") })
        assertFalse(messages.any { it.content.contains("hidden thought") || it.content.contains("TOOL_RESULT") })
        assertEquals(3, history.size)
    }

    private fun ui(chat: AgentChatIntegration): Pair<ChatActivity, ChatPresenter> {
        val controller = Robolectric.buildActivity(ChatActivity::class.java).create()
        activities.add(controller)
        val activity = controller.get()
        activity.modelName = "Qwen3.5-0.8B"
        field(activity, "layoutModelLoading", activity.findViewById<android.view.View>(R.id.layout_model_loading))
        val presenter = mockk<ChatPresenter>(relaxed = true)
        every { presenter.isAgentMode } returns true
        every { presenter.agentChat } returns chat
        every { presenter.contextUsage } returns MutableStateFlow(ContextUsageState())
        field(activity, "chatPresenter", presenter)
        field(activity, "chatInputModule", mockk<ChatInputComponent>(relaxed = true))
        val item = assistant("").apply { loading = true }
        activity.chatListComponent = mockk<ChatListComponent>(relaxed = true)
        every { activity.chatListComponent.recentItem } returns item
        return activity to presenter
    }

    private fun field(target: Any, name: String, value: Any) {
        val owner = if (target is ChatPresenter) ChatPresenter::class.java else ChatActivity::class.java
        owner.getDeclaredField(name).apply { isAccessible = true }.set(target, value)
    }

    private fun readIncarnation(p: ChatPresenter): Int =
        ChatPresenter::class.java.getDeclaredField("incarnation").apply { isAccessible = true }.getInt(p)

    private fun bumpIncarnation(p: ChatPresenter) {
        ChatPresenter::class.java.getDeclaredMethod("bumpIncarnation").apply { isAccessible = true }.invoke(p)
    }

    private fun isCurrent(p: ChatPresenter, captured: Int): Boolean =
        ChatPresenter::class.java.getDeclaredMethod("isCurrent", Int::class.javaPrimitiveType)
            .apply { isAccessible = true }.invoke(p, captured) as Boolean

    @Test
    fun capturedIncarnationGoesStaleAfterConversationReplaced() {
        val presenter = ChatPresenter(presenterActivity(), "Qwen3.5-0.8B", "test")
        val captured = readIncarnation(presenter)
        assertTrue("a fresh turn matches the current conversation", isCurrent(presenter, captured))
        bumpIncarnation(presenter)
        assertFalse("replacing the conversation must invalidate the captured turn", isCurrent(presenter, captured))
        assertTrue("a turn captured after the bump is current again", isCurrent(presenter, readIncarnation(presenter)))
    }

    @Test
    fun destroyBumpsIncarnationSoAnInFlightTurnIsDropped() {
        val presenter = ChatPresenter(presenterActivity(), "Qwen3.5-0.8B", "test")
        val captured = readIncarnation(presenter)
        presenter.destroy()
        assertTrue("teardown must advance the incarnation", readIncarnation(presenter) > captured)
        assertFalse("a turn captured before teardown is no longer current", isCurrent(presenter, captured))
    }

    @Test
    fun realRunnerFeedsToolStatesAndThinkingUi_activityDoesNotWriteAgain() = runTest {
        Dispatchers.setMain(UnconfinedTestDispatcher(testScheduler))
        val chat = integration(StandardTestDispatcher(testScheduler))
        chat.changeMode("history", true)
        val (activity, presenter) = ui(chat)
        activity.onGenerateStart(user("2*3"))
        val inference = FakeInference("<think>calculate</think><tool_call><function=calculator><parameter=expression>2*3</parameter></function></tool_call>",
            "<answer>**6**</answer>")
        val updates = mutableListOf<AgentRunState>()
        val final = chat.run(inference, AgentRequest("2*3", sessionId = "history"),
            AgentToolExecutor { _, _ -> AgentToolResult("[CALC_RESULT]\nresult: 6") }) {
            updates.add(it)
            activity.onAgentState(it)
        }
        assertTrue(updates.any { state -> state.steps.any { it.status == AgentStepStatus.RUNNING } })
        assertFalse(updates.any { it.answer.contains("tool_call") || it.answer.contains("function=") })
        activity.onGenerateFinished(hashMapOf("agent" to true, "agent_state" to final))
        val item = activity.chatListComponent.recentItem!!
        assertEquals("**6**", item.displayText)
        assertEquals("calculate", item.thinkingText)
        assertEquals(ChatViewHolders.THINKING_COMPLETE_WITHOUT_TIMING, item.thinkingFinishedTime)
        val steps = activity.findViewById<TextView>(R.id.tv_agent_steps).text.toString()
        assertTrue(steps.contains(activity.getString(R.string.agent_tool_calculator)))
        assertTrue(steps.contains(activity.getString(R.string.agent_step_done)))
        verify(exactly = 0) { presenter.saveResponseToDatabase(any()) }
        assertFalse(activity.isGenerating)
    }

    @Test
    fun finalAnswerUsesExistingAnswerFilterOnceInUiAndHistory() = runTest {
        Dispatchers.setMain(UnconfinedTestDispatcher(testScheduler))
        val chat = integration(StandardTestDispatcher(testScheduler))
        val (activity, presenter) = ui(chat)
        activity.onGenerateStart(user("show literal tags"))
        val answer = "<answer><answer>literal</answer></answer>"
        activity.onGenerateFinished(hashMapOf("agent" to true,
            "agent_state" to AgentRunState(phase = AgentPhase.COMPLETED, answer = answer)))
        verify(exactly = 0) { presenter.saveResponseToDatabase(any()) }
        val item = activity.chatListComponent.recentItem!!
        assertEquals("<answer>literal</answer>", item.displayText)
        assertEquals("<answer>literal</answer>", AgentChatIntegration.history(listOf(item)).single().content)
    }

    @Test
    fun failedOrStoppedAgentTurns_haveNoNativeMetricsRequirementAndAreNotSaved() = runTest {
        Dispatchers.setMain(UnconfinedTestDispatcher(testScheduler))
        val chat = integration(StandardTestDispatcher(testScheduler))
        val (activity, presenter) = ui(chat)
        activity.onGenerateStart(user("request"))
        activity.onGenerateFinished(hashMapOf("agent" to true,
            "agent_state" to AgentRunState(phase = AgentPhase.CANCELLED, answer = "partial")))
        verify(exactly = 0) { presenter.saveResponseToDatabase(any()) }
        assertFalse(activity.isGenerating)
        activity.onGenerateStart(user("retry"))
        activity.onGenerateFinished(hashMapOf("agent" to true, "error" to true))
        assertEquals(activity.getString(R.string.agent_failed), activity.chatListComponent.recentItem!!.displayText)
        verify(exactly = 0) { presenter.saveResponseToDatabase(any()) }
    }

    @Test
    fun consentRevocationCancelsFakeInferenceAndCannotPersist() = runTest {
        val chat = integration(StandardTestDispatcher(testScheduler))
        chat.changeMode("history", true)
        val entered = CompletableDeferred<Unit>()
        var calls = 0
        val inference = object : AgentInference {
            override suspend fun countTokens(messages: List<AgentMessage>) = 100
            override suspend fun generate(request: AgentInferenceRequest, onDelta: (String) -> Unit): AgentGeneration {
                calls++
                entered.complete(Unit)
                awaitCancellation()
            }
        }
        val job = launch { chat.run(inference, AgentRequest("记住：我喜欢茶", sessionId = "history"), noTools) }
        entered.await()
        chat.close()
        job.join()
        assertTrue(job.isCancelled)
        assertEquals(1, calls)
        assertEquals(AgentPhase.CANCELLED, chat.state.value.phase)
        assertFalse(chat.mode.value.authorized)
    }

    @Test
    fun cancellationWaitsForBlockingInferenceToExitBeforeCompletion() = runBlocking {
        val chat = integration(Dispatchers.IO)
        chat.changeMode("history", true)
        val entered = CountDownLatch(1)
        val exit = CountDownLatch(1)
        val cancelled = CountDownLatch(1)
        val inference = AgentBlockingInference(submit = { _, _ ->
            entered.countDown()
            assertTrue(exit.await(3, TimeUnit.SECONDS))
            emptyMap()
        }, cancel = { cancelled.countDown() })
        val job = launch(Dispatchers.IO) {
            chat.run(inference, AgentRequest("hello", sessionId = "history"), noTools)
        }
        try {
            assertTrue(entered.await(3, TimeUnit.SECONDS))
            job.cancel()
            assertTrue(cancelled.await(3, TimeUnit.SECONDS))
            assertFalse(job.isCompleted)
            exit.countDown()
            job.join()
            assertEquals(AgentPhase.CANCELLED, chat.state.value.phase)
        } finally { exit.countDown(); job.cancelAndJoin() }
    }

    @Test
    fun presenterExitWaitsForAgentInferenceBeforeReleasingItsSession() = runBlocking {
        val chat = integration(Dispatchers.IO)
        chat.changeMode("history", true)
        val activity = mockk<ChatActivity>(relaxed = true)
        mockkObject(ChatDataManager.Companion)
        every { ChatDataManager.getInstance(any()) } returns mockk(relaxed = true)
        val presenter = ChatPresenter(activity, "test", "test", chat)
        val session = mockk<ChatSession>(relaxed = true)
        field(presenter, "chatSession", session)
        val entered = CountDownLatch(1)
        val exit = CountDownLatch(1)
        val cancelled = CountDownLatch(1)
        val inference = AgentBlockingInference(submit = { _, _ ->
            entered.countDown()
            assertTrue(exit.await(3, TimeUnit.SECONDS))
            emptyMap()
        }, cancel = { cancelled.countDown() })
        val job = launch(Dispatchers.IO) { chat.run(inference, AgentRequest("hello", sessionId = "history"), noTools) }
        field(presenter, "activeGeneration", job)
        try {
            assertTrue(entered.await(3, TimeUnit.SECONDS))
            presenter.destroy()
            assertTrue(cancelled.await(3, TimeUnit.SECONDS))
            verify(exactly = 0) { session.release() }
            exit.countDown()
            job.join()
            verify(timeout = 3000, exactly = 1) { session.release() }
        } finally { exit.countDown(); job.cancelAndJoin() }
    }

    @Test
    fun textOnlyGateRejectsMediaBeforeStorageOrAnyTools() = runTest {
        Dispatchers.setMain(UnconfinedTestDispatcher(testScheduler))
        val chat = integration(StandardTestDispatcher(testScheduler))
        chat.changeMode("history", true)
        val activity = mockk<ChatActivity>(relaxed = true)
        mockkObject(ChatDataManager.Companion)
        val data = mockk<ChatDataManager>(relaxed = true)
        every { ChatDataManager.getInstance(any()) } returns data
        val presenter = ChatPresenter(activity, "test", "test", chat)
        val listener = mockk<ChatPresenter.GenerateListener>(relaxed = true)
        val media = listOf(user("image").apply { imageUris = listOf(Uri.parse("content://test/image")) },
            user("audio").apply { audioUri = Uri.parse("content://test/audio") },
            user("video").apply { videoUri = Uri.parse("content://test/video") },
            user("<img>/local/image</img>"), user("<audio>/local/audio</audio>"))
        media.forEach { assertEquals(R.string.agent_text_only, presenter.requestGenerate(it, listener)["agent_rejected"]) }
        chat.restore("history", emptyList())
        assertEquals(R.string.agent_consent_required,
            presenter.requestGenerate(user("no authorization"), listener)["agent_rejected"])
        verify(exactly = 0) { data.addChatData(any(), any()) }
        verify(exactly = 0) { listener.onGenerateStart() }
    }

    @Test
    fun ordinaryPresenterUsesExistingGenerationPathWithoutAgentTools() = runTest {
        Dispatchers.setMain(UnconfinedTestDispatcher(testScheduler))
        val chat = integration(StandardTestDispatcher(testScheduler))
        val activity = mockk<ChatActivity>(relaxed = true)
        val lifecycle = LifecycleRegistry.createUnsafe(activity)
        every { activity.lifecycle } returns lifecycle
        lifecycle.currentState = Lifecycle.State.RESUMED
        mockkObject(ChatDataManager.Companion)
        every { ChatDataManager.getInstance(any()) } returns mockk(relaxed = true)
        val presenter = ChatPresenter(activity, "test", "test", chat)
        val session = mockk<ChatSession>(relaxed = true)
        every { session.generate(any(), any(), any()) } returns hashMapOf("response" to "normal")
        field(presenter, "chatSession", session)
        field(presenter, "sessionId", "ordinary")
        val result = presenter.requestGenerate(user("normal user"), mockk(relaxed = true))
        assertNull(result["agent"])
        verify(exactly = 1) { session.generate("normal user", any(), any()) }
        assertFalse(chat.mode.value.agent)
        assertTrue(chat.mode.value.locked)
        lifecycle.currentState = Lifecycle.State.DESTROYED
    }

    @Test
    fun destroyMarksStaleLoaderBeforeItCanReleaseTheNewResident() = runBlocking {
        Dispatchers.setMain(UnconfinedTestDispatcher())
        val service = ChatService()
        val oldReleaseStarted = CountDownLatch(1)
        val stopped = CountDownLatch(1)
        val marker = CountDownLatch(1)
        val newReleased = AtomicBoolean(false)
        val previous = FakeSession("previous", releaser = { oldReleaseStarted.countDown() })
        val next = FakeSession("new-resident", releaser = { newReleased.set(true) })
        val stale = FakeSession("stale", loader = { session ->
            try { service.beforeLoad(session) { session.releaseRequested.get() } }
            finally { stopped.countDown() }
        }, releaser = { marker.countDown(); service.onNativeReleased(it, true) })
        service.beforeLoad(previous)
        val presenter = ChatPresenter(presenterActivity(), "Qwen3.5-0.8B", "test", integration(Dispatchers.IO))
        field(presenter, "chatSession", stale)
        presenter.load()
        try {
            assertTrue(oldReleaseStarted.await(3, TimeUnit.SECONDS))
            presenter.destroy()
            assertEquals(0L, marker.count)
            assertTrue(stale.releaseRequested.get())
            service.onNativeReleased(previous, true)
            service.beforeLoad(next)
            assertTrue(stopped.await(3, TimeUnit.SECONDS))
            assertFalse(newReleased.get())
        } finally { service.onNativeReleased(previous, true); service.onNativeReleased(next, true) }
    }

    @Implements(LlmSession::class, isInAndroidSdk = false)
    class NativeFreeLlmSession {
        companion object {
            @JvmStatic
            @Implementation
            fun __staticInitializer__() { }
        }
    }

    @Test
    @Config(shadows = [NativeFreeLlmSession::class])
    fun oldPresenterDestroyCannotReleaseSharedSessionAfterRuntimeOwnerTransfer() = runBlocking {
        Dispatchers.setMain(UnconfinedTestDispatcher())
        val session = mockk<LlmSession>(relaxed = true)
        val ready = AtomicBoolean(true)
        val entered = CountDownLatch(1)
        val finish = CountDownLatch(1)
        val oldCleanup = CountDownLatch(1)
        val released = CountDownLatch(1)
        every { session.sessionId } returns "shared-session"
        every { session.getHistory() } returns emptyList()
        every { session.isModelLoaded() } answers { ready.get() }
        every { session.load() } answers {
            entered.countDown()
            assertTrue(finish.await(3, TimeUnit.SECONDS))
        }
        every { session.release() } answers { released.countDown() }
        val runtime = DefaultLlmRuntimeController
        val fields = listOf("activeSession", "activeOwner", "activeModelId", "activeUseAppConfig").associateWith {
            runtime.javaClass.getDeclaredField(it).apply { isAccessible = true }
        }
        val saved = fields.mapValues { it.value.get(runtime) }
        mockkObject(ServiceLocator, DefaultLlmRuntimeController, ServerEventManager.Companion)
        every { ServiceLocator.getLlmRuntimeController() } returns runtime
        every { ServerEventManager.getInstance() } returns ServerEventManager()
        fields.getValue("activeSession").set(runtime, session)
        fields.getValue("activeOwner").set(runtime, null)
        fields.getValue("activeModelId").set(runtime, "Qwen3.5-0.8B")
        fields.getValue("activeUseAppConfig").set(runtime, true)
        val old = ChatPresenter(presenterActivity(), "Qwen3.5-0.8B", "test")
        val next = ChatPresenter(presenterActivity(), "Qwen3.5-0.8B", "test")
        val ownerField = ChatPresenter::class.java.getDeclaredField("sessionOwner").apply { isAccessible = true }
        val oldOwner = ownerField.get(old)
        val nextOwner = ownerField.get(next)
        var loading: Job? = null
        every { runtime.releaseSession(any(), any()) } answers {
            val expected = firstArg<LlmSession?>()
            val owner = secondArg<Any?>()
            callOriginal()
            if (expected === session && owner === oldOwner && loading?.isCompleted == true) oldCleanup.countDown()
        }
        try {
            assertSame(session, old.createSession())
            ready.set(false)
            old.load()
            loading = ChatPresenter::class.java.getDeclaredField("loadJob").apply { isAccessible = true }.get(old) as Job
            assertTrue(entered.await(3, TimeUnit.SECONDS))
            ready.set(true)
            assertSame(session, next.createSession())
            runtime.releaseSession(mockk<LlmSession>(relaxed = true), nextOwner)
            old.destroy()
            verify { runtime.releaseSession(session, oldOwner) }
            verify(exactly = 0) { session.release() }
            finish.countDown()
            loading?.join()
            assertTrue(oldCleanup.await(3, TimeUnit.SECONDS))
            verify(exactly = 0) { session.release() }
            next.destroy()
            assertTrue(released.await(3, TimeUnit.SECONDS))
            verify { runtime.releaseSession(session, nextOwner) }
            verify(exactly = 1) { session.release() }
        } finally {
            finish.countDown()
            loading?.join()
            fields.forEach { (name, field) -> field.set(runtime, saved[name]) }
        }
    }

    @Test
    fun actualRuntimeHistoryLocksModeEvenWhenIntentHasNoSessionId() = runTest {
        val chat = integration(StandardTestDispatcher(testScheduler))
        val runtimeHistory = listOf(user("existing runtime user"), assistant("existing answer"))
        val runtime = FakeSession("actual-runtime-id", runtimeHistory)
        val activity = presenterActivity()
        val receivedHistories = mutableListOf<List<ChatDataItem>?>()
        val presenter = ChatPresenter(activity, "Qwen3.5-0.8B", "test", chat,
            sessionFactory = { agent, _, history, _ ->
                assertFalse(agent)
                receivedHistories.add(history)
                runtime
            })
        presenter.restoreAgentMode()
        assertTrue(chat.mode.value.canChange)
        assertSame(runtime, presenter.createSession())
        assertFalse(chat.mode.value.canChange)
        assertSame(runtime, presenter.restoreCreatedSessionMode())
        assertEquals("actual-runtime-id", presenter.getSessionId())
        assertEquals(runtimeHistory, presenter.getChatHistory())
        rejected { presenter.changeAgentMode(true) }
        assertEquals(listOf<List<ChatDataItem>?>(null), receivedHistories)
        assertFalse(runtime.releaseRequested.get())
        assertFalse(chat.store.isAgentSession("actual-runtime-id"))
    }

    @Test
    fun actualIdRestoresAgentStatusAndUsesAnExclusiveSessionWithItsHistory() = runTest {
        val chat = integration(StandardTestDispatcher(testScheduler))
        chat.changeMode("actual-agent-id", true)
        val history = listOf(user("previous Agent user"), assistant("previous answer"))
        val reused = FakeSession("actual-agent-id", history)
        val presenter = ChatPresenter(presenterActivity(), "Qwen3.5-0.8B", "test", chat,
            sessionFactory = { agent, id, items, _ -> if (agent) FakeSession(id!!, items.orEmpty()) else reused })
        presenter.restoreAgentMode()
        presenter.createSession()
        val actual = presenter.restoreCreatedSessionMode()
        assertNotSame(reused, actual)
        assertTrue(reused.releaseRequested.get())
        assertEquals(history, actual.getHistory())
        assertEquals(history, presenter.getChatHistory())
        assertTrue(presenter.isAgentMode)
        assertFalse(chat.mode.value.authorized)
        assertFalse(chat.mode.value.canChange)
    }

    @Test
    fun failedNormalReplacementIsClosedBeforeAnotherMessageCanBeStored() = runTest {
        val chat = integration(StandardTestDispatcher(testScheduler))
        chat.changeMode("session", true)
        val old = FakeSession("session")
        val activity = presenterActivity()
        activity.intent.putExtra("chatSessionId", "session")
        val presenter = ChatPresenter(activity, "Qwen3.5-0.8B", "test", chat,
            sessionFactory = { agent, _, _, _ ->
                if (!agent) throw IllegalStateException("injected ensureSession failure")
                old
            })
        presenter.createSession()
        rejected { presenter.changeAgentMode(false) }
        assertTrue(old.releaseRequested.get())
        assertFalse(presenter.isAgentMode)
        val listener = mockk<ChatPresenter.GenerateListener>(relaxed = true)
        assertEquals(true, presenter.requestGenerate(user("must not send"), listener)["session_unavailable"])
        verify(exactly = 0) { listener.onGenerateStart() }
        val data = ChatDataManager.getInstance(activity)
        verify(exactly = 0) { data.addChatData(any(), any()) }
    }

    @Test
    fun leavingAgentTwice_restoresContextUsageWithoutAnotherEmissionOrMessage() = runTest {
        Dispatchers.setMain(UnconfinedTestDispatcher(testScheduler))
        val chat = integration(StandardTestDispatcher(testScheduler))
        val (activity, presenter) = ui(chat)
        every { presenter.isAgentMode } answers { chat.mode.value.agent }
        field(activity, "sessionId", "history")
        val state = ContextUsageState(tokens = 18, limit = 8192)
        val usage = MutableStateFlow(state)
        every { presenter.contextUsage } returns usage
        var emissions = 0
        backgroundScope.launch(UnconfinedTestDispatcher(testScheduler)) {
            usage.collect { emissions++ }
        }
        val counter = activity.findViewById<TextView>(R.id.tv_context_usage)
        val expected = activity.getString(R.string.context_usage_format, "18", "8K")
        ChatActivity::class.java.getDeclaredMethod("renderContextUsage", ContextUsageState::class.java)
            .apply { isAccessible = true }.invoke(activity, state)
        assertEquals(View.VISIBLE, counter.visibility)
        assertEquals(expected, counter.text.toString())
        repeat(2) {
            chat.changeMode("history", true)
            activity.onLoadingChanged(false)
            assertEquals(View.GONE, counter.visibility)
            chat.changeMode("history", false)
            usage.value = state.copy()
            activity.onLoadingChanged(false)
            assertEquals(1, emissions)
            assertEquals(View.VISIBLE, counter.visibility)
            assertEquals(expected, counter.text.toString())
            assertEquals(activity.getString(R.string.context_usage_description, 18, 8192), counter.contentDescription)
            assertEquals(state, usage.value)
        }
        verify(exactly = 0) { presenter.refreshContextUsage() }
        verify(exactly = 0) { presenter.load() }
    }

    @Test
    fun leavingAgent_preservesHiddenCompressingAndNearlyFullContextStates() = runTest {
        Dispatchers.setMain(UnconfinedTestDispatcher(testScheduler))
        val chat = integration(StandardTestDispatcher(testScheduler))
        val (activity, presenter) = ui(chat)
        every { presenter.isAgentMode } answers { chat.mode.value.agent }
        val usage = MutableStateFlow(ContextUsageState())
        every { presenter.contextUsage } returns usage
        val counter = activity.findViewById<TextView>(R.id.tv_context_usage)
        for (state in listOf(
            ContextUsageState(),
            ContextUsageState(limit = 8192),
            ContextUsageState(compressing = true),
            ContextUsageState(tokens = 7000, limit = 8192)
        )) {
            chat.changeMode("history", true)
            usage.value = state
            activity.onLoadingChanged(false)
            assertEquals(View.GONE, counter.visibility)
            chat.changeMode("history", false)
            activity.onLoadingChanged(false)
            assertEquals(if (state.visible || state.compressing) View.VISIBLE else View.GONE, counter.visibility)
            if (state.compressing) {
                assertEquals(activity.getString(R.string.context_compressing), counter.text.toString())
            } else if (state.visible) {
                assertEquals(activity.getString(R.string.context_usage_format,
                    ContextUsageFormat.tokens(state.tokens), ContextUsageFormat.tokens(state.limit)), counter.text.toString())
            }
            if (state.nearlyFull) {
                assertEquals(com.google.android.material.color.MaterialColors.getColor(
                    counter, androidx.appcompat.R.attr.colorError, counter.currentTextColor), counter.currentTextColor)
            }
            assertEquals(state, usage.value)
        }
    }

    @Test
    fun uiNormalReplacementFailureFinishesInsteadOfRestoringInput() = runTest {
        Dispatchers.setMain(UnconfinedTestDispatcher(testScheduler))
        val (original, presenter) = ui(integration(StandardTestDispatcher(testScheduler)))
        val activity = spyk(original)
        every { presenter.isAgentMode } returns false
        coEvery { presenter.changeAgentMode(false) } throws IllegalStateException("injected replacement failure")
        ChatActivity::class.java.getDeclaredMethod("changeAgentMode", java.lang.Boolean.TYPE)
            .apply { isAccessible = true }.invoke(activity, false)
        verify { activity.finish() }
        verify(exactly = 0) { presenter.load() }
        assertTrue(activity.isLoading)
        assertFalse(activity.findViewById<View>(R.id.btn_send).isEnabled)
    }

    @Test
    fun uiAgentFailureKeepsRetryButCannotReenableSending() = runTest {
        Dispatchers.setMain(UnconfinedTestDispatcher(testScheduler))
        val (original, _) = ui(integration(StandardTestDispatcher(testScheduler)))
        val activity = spyk(original)
        activity.onModelLoadFailed("injected Agent failure")
        assertTrue(activity.isLoading)
        assertFalse(activity.findViewById<View>(R.id.btn_send).isEnabled)
        val menu = MenuBuilder(activity)
        activity.menuInflater.inflate(R.menu.menu_chat, menu)
        activity.onPrepareOptionsMenu(menu)
        assertTrue(menu.findItem(R.id.agent_retry).isVisible)
        verify(exactly = 0) { activity.finish() }
    }

    @Test
    fun finalPersistenceUsesIoAndFrozenSessionIdOnce_evenIfUiFinishesTwice() = runTest {
        Dispatchers.setMain(UnconfinedTestDispatcher(testScheduler))
        val chat = integration(StandardTestDispatcher(testScheduler))
        chat.changeMode("original-session", true)
        val activity = presenterActivity()
        val data = mockk<ChatDataManager>(relaxed = true)
        every { ChatDataManager.getInstance(any()) } returns data
        every { data.getChatDataBySession(any()) } returns emptyList()
        val saved = slot<ChatDataItem>()
        every { data.addChatData("original-session", capture(saved)) } answers {
            assertNotSame(Looper.getMainLooper().thread, Thread.currentThread())
            saved.captured.dbId = 42
        }
        val presenter = ChatPresenter(activity, "Qwen3.5-0.8B", "test", chat)
        field(presenter, "sessionId", "different-session")
        val state = AgentRunState(phase = AgentPhase.COMPLETED, answer = "<answer><answer>literal</answer></answer>",
            thinking = "private thought", steps = listOf(AgentStep(0, AgentStepKind.SEARCH, "page data",
                detail = "private query", status = AgentStepStatus.DONE, result = "full webpage data")))
        val snapshot = AgentTurnSnapshot("original-session", "2026-11-23T12:00:00Z", "12:00", chat.consent.state.value.revision, state)
        assertEquals(42L, presenter.persistAgentTurn(snapshot).assistantId)
        assertEquals(42L, presenter.persistAgentTurn(snapshot).assistantId)
        val (uiActivity, uiPresenter) = ui(chat)
        val result = hashMapOf<String, Any>("agent" to true, "agent_state" to state, "agent_assistant_id" to 42L)
        uiActivity.onGenerateStart(user("query"))
        uiActivity.onGenerateFinished(result)
        uiActivity.onGenerateFinished(result)
        verify(exactly = 1) { data.addChatData("original-session", any()) }
        verify(exactly = 0) { data.addChatData("different-session", any()) }
        verify(exactly = 0) { uiPresenter.saveResponseToDatabase(any()) }
        assertNull(saved.captured.thinkingText)
        assertEquals("<answer>literal</answer>", saved.captured.displayText)
        assertEquals("<answer>literal</answer>", AgentChatIntegration.history(listOf(saved.captured)).single().content)
        val record = chat.store.toolStepsForSession("original-session").single()
        assertEquals(42L, record.assistantId)
        assertFalse(record.toString().contains("private query"))
        assertFalse(record.toString().contains("full webpage data"))
    }

    @Test
    fun sqliteToolStepsReopenWithoutConsentAndNeverBecomeInferenceHistory() = runTest {
        val dispatcher = StandardTestDispatcher(testScheduler)
        val context = ApplicationProvider.getApplicationContext<Application>()
        val name = "agent-reopen-${UUID.randomUUID()}.sqlite3"
        val consent = AgentConsentController(dispatcher)
        val store = SqliteAgentStore(context, consent, databaseName = name)
        stores.add(store)
        val grant = consent.confirmEnable(networkAllowed = false, persistenceAllowed = false)
        store.setAgentSession("conversation", true, grant.revision)
        val record = AgentToolStepRecord(7, "2026-11-23T12:00:00Z", AgentPhase.COMPLETED,
            listOf(AgentStoredStep(AgentStepKind.READ_PAGE, "arbitrary webpage", "<tool_call>private data</tool_call>", AgentStepStatus.DONE)))
        assertTrue(store.saveToolSteps("conversation", record, grant.revision))
        assertFalse(store.saveMemory("must not persist memory", "test", grant.revision))
        store.close()
        val freshConsent = AgentConsentController(dispatcher)
        val reopened = SqliteAgentStore(context, freshConsent, databaseName = name)
        stores.add(reopened)
        val restored = reopened.toolStepsForSession("conversation").single()
        assertFalse(reopened.saveToolSteps("conversation", record, freshConsent.state.value.revision))
        assertEquals(7L, restored.assistantId)
        assertEquals("read_webpage", restored.steps.single().tool)
        assertEquals(AgentStepStatus.DONE, restored.steps.single().status)
        assertFalse(restored.toString().contains("private data"))
        assertFalse(restored.toString().contains("arbitrary webpage"))
        val spy = spyk(reopened)
        val chat = AgentChatIntegration({ spy }, freshConsent, dispatcher)
        chat.restore("conversation", listOf(user("previous")))
        assertFalse(chat.mode.value.authorized)
        chat.confirm()
        val inference = FakeInference("model thought</think><answer>final</answer>")
        chat.run(inference, AgentRequest("current", history = listOf(AgentMessage(AgentRole.USER, "previous")),
            expectsThinking = true, sessionId = "conversation"), noTools)
        assertTrue(inference.requests.single().allowThinking)
        coVerify(exactly = 0) { spy.toolStepsForSession(any()) }
        assertFalse(inference.requests.single().messages.any { it.content.contains(record.time) || it.content.contains("assistantId") })
        assertEquals(1, reopened.toolStepsForSession("conversation").size)
    }

    @Test
    fun sqliteVersionOneUpgradeAddsToolStepsWithoutClearingSharedData() = runTest {
        val context = ApplicationProvider.getApplicationContext<Application>()
        val name = "agent-migration-${UUID.randomUUID()}.sqlite3"
        val legacy = object : SQLiteOpenHelper(context, name, null, 1) {
            override fun onCreate(db: SQLiteDatabase) {
                db.execSQL("CREATE TABLE agentSessions(historyId TEXT PRIMARY KEY NOT NULL)")
                db.execSQL("CREATE TABLE agentMemories(content TEXT PRIMARY KEY NOT NULL,category TEXT NOT NULL,updatedAt INTEGER NOT NULL)")
                db.execSQL("CREATE TABLE agentSkills(name TEXT PRIMARY KEY NOT NULL,triggers TEXT NOT NULL,actionTemplate TEXT NOT NULL,updatedAt INTEGER NOT NULL)")
                db.execSQL("INSERT INTO agentMemories VALUES('keep tea','test',1)")
            }
            override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) { }
        }
        legacy.writableDatabase
        legacy.close()
        val consent = AgentConsentController(StandardTestDispatcher(testScheduler))
        val store = SqliteAgentStore(context, consent, databaseName = name)
        stores.add(store)
        assertEquals("keep tea", store.allMemories().single().content)
        val grant = consent.confirmEnable()
        val record = AgentToolStepRecord(-1, "2026-11-23T12:00:00Z", AgentPhase.CANCELLED,
            listOf(AgentStoredStep.from(AgentStep(0, AgentStepKind.SEARCH, "", status = AgentStepStatus.CANCELLED))))
        assertTrue(store.saveToolSteps("conversation", record, grant.revision))
        assertEquals(record, store.toolStepsForSession("conversation").single())
        store.clearToolSteps("conversation")
        assertTrue(store.toolStepsForSession("conversation").isEmpty())
        assertEquals("keep tea", store.allMemories().single().content)
    }

    @Test
    fun localSkillsAndMemoriesAreManagedIndependentlyAndSharedAcrossSessions() = runTest {
        val chat = integration(StandardTestDispatcher(testScheduler))
        chat.changeMode("history", true)
        val grant = chat.consent.state.value.revision
        chat.store.saveMemory("likes tea", "preference", grant)
        chat.store.saveMemory("likes coffee", "preference", grant)
        chat.store.saveSkill(AgentSkill("release", listOf("release"), "test then release"), grant)
        chat.store.deleteMemory("likes coffee")
        assertEquals("likes tea", chat.store.allMemories().single().content)
        chat.store.clearSkills()
        assertTrue(chat.store.allSkills().isEmpty())
        assertEquals(1, chat.store.allMemories().size)
        chat.restore("another conversation", emptyList())
        assertFalse(chat.mode.value.agent)
        assertEquals(1, chat.store.allMemories().size)
        chat.store.clearAll()
        assertTrue(chat.store.allMemories().isEmpty())
        assertTrue(chat.store.isAgentSession("history"))
    }
}
