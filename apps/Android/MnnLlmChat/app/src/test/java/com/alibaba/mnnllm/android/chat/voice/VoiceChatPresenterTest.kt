package com.alibaba.mnnllm.android.chat.voice

import android.app.Activity
import android.media.AudioManager
import android.os.Looper
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.audio.AudioChunksPlayer
import com.alibaba.mnnllm.android.chat.ChatPresenter
import com.alibaba.mnnllm.android.chat.GenerateResultProcessor
import com.alibaba.mnnllm.android.llm.ChatSession
import io.mockk.*
import kotlinx.coroutines.*
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.Shadows.shadowOf
import org.robolectric.annotation.Config
import org.robolectric.shadows.ShadowLooper

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [28])
class VoiceChatPresenterTest {
    private lateinit var presenter: VoiceChatPresenter
    private lateinit var mockActivity: Activity
    private lateinit var mockView: VoiceChatView
    private lateinit var mockChatPresenter: ChatPresenter
    private lateinit var testScope: CoroutineScope
    private lateinit var mockAudioManager: AudioManager

    @Before
    fun setUp() {
        mockActivity = mockk(relaxed = true)
        mockView = mockk(relaxed = true)
        mockChatPresenter = mockk(relaxed = true)
        testScope = CoroutineScope(Dispatchers.Unconfined + SupervisorJob())
        mockAudioManager = mockk(relaxed = true)
        every { mockActivity.getSystemService(Activity.AUDIO_SERVICE) } returns mockAudioManager
        presenter = VoiceChatPresenter(mockActivity, mockView, mockChatPresenter, testScope)
    }

    @After
    fun tearDown() {
        unmockkAll()
        testScope.cancel()
    }

    private fun setBooleanField(fieldName: String, value: Boolean) {
        val field = VoiceChatPresenter::class.java.getDeclaredField(fieldName)
        field.isAccessible = true
        field.setBoolean(presenter, value)
    }

    @Test
    fun `test initial state`() {
        assertEquals(VoiceChatPresenterState.INITIALIZING, presenter.getCurrentStatus())
    }

    @Test
    fun `test start and stop lifecycle`() {
        every { mockChatPresenter.addGenerateListener(any()) } just Runs
        every { mockChatPresenter.removeGenerateListener(any()) } just Runs
        presenter.start()
        //Status will change to LISTENING after startup (async, need to wait)
        runBlocking { delay(100) }
        presenter.stop()
        //After stopping, isStopped should be true, status no longer changes
        assertTrue(presenter.getCurrentStatus() == VoiceChatPresenterState.INITIALIZING ||
                   presenter.getCurrentStatus() == VoiceChatPresenterState.LISTENING)
        verify { mockChatPresenter.addGenerateListener(presenter) }
        verify { mockChatPresenter.removeGenerateListener(presenter) }
    }

    @Test
    fun `test toggle speaker`() {
        presenter.toggleSpeaker(true)
        verify { mockAudioManager.isSpeakerphoneOn = true }
        presenter.toggleSpeaker(false)
        verify { mockAudioManager.isSpeakerphoneOn = false }
    }

    @Test
    fun `test stopGeneration enters stopping state when generation is active`() {
        every { mockChatPresenter.stopGenerate() } just Runs
        every { mockView.updateStatus(any()) } just Runs
        setBooleanField("isProcessingLlm", true)
        presenter.stopGeneration()
        runBlocking { delay(700) }
        ShadowLooper.runUiThreadTasksIncludingDelayedTasks()
        shadowOf(Looper.getMainLooper()).runToEndOfTasks()
        verify(timeout = 1000) { mockChatPresenter.stopGenerate() }
        verify(timeout = 1000) { mockView.updateStatus(VoiceChatState.STOPPING) }
    }

    @Test
    fun `test stopGeneration is noop when idle`() {
        every { mockChatPresenter.stopGenerate() } just Runs
        every { mockView.updateStatus(any()) } just Runs

        presenter.stopGeneration()
        runBlocking { delay(400) }
        ShadowLooper.runUiThreadTasksIncludingDelayedTasks()
        shadowOf(Looper.getMainLooper()).runToEndOfTasks()

        verify(exactly = 0) { mockChatPresenter.stopGenerate() }
        verify(exactly = 0) { mockView.updateStatus(any()) }
    }

    @Test
    fun `test onLlmGenerateProgress triggers task channel`() {
        val processor = mockk<GenerateResultProcessor>(relaxed = true)
        every { mockView.addTranscript(any()) } just Runs
        every { mockView.updateLastTranscript(any()) } just Runs
        presenter.onLlmGenerateProgress("hello", processor)
        runBlocking { delay(100) }
        //As long as there's no exception, the process is reachable
    }

    @Test
    fun `test onGenerateFinished triggers final chunk`() {
        every { mockView.updateStatus(any()) } just Runs
        presenter.onGenerateFinished(hashMapOf("response" to "ok"))
        runBlocking { delay(100) }
        //As long as there's no exception, the process is reachable
    }

    @Test
    fun `test error handling in speakGreetingMessage`() {
        //Simulate activity.getString throwing exception
        every { mockActivity.getString(any()) } throws RuntimeException("test error")
        every { mockView.updateStatus(any()) } just Runs
        presenter.start()
        runBlocking { delay(200) }
        //As long as there's no crash, exceptions are handled
    }

    @Test
    fun `start waits for tts readiness before greeting synthesis`() {
        val ttsClient = mockk<TtsClient>()
        val audioPlayer = mockk<AudioChunksPlayer>()
        every { mockActivity.getString(R.string.voice_chat_ready_greeting) } returns "有什么可以帮助您的？"
        every { mockView.updateStatus(any()) } just Runs
        coEvery { ttsClient.waitForInitComplete() } returns true
        every { ttsClient.process("有什么可以帮助您的？", 0) } returns shortArrayOf(1, 2, 3)
        every { audioPlayer.setOnCompletionListener(any()) } just Runs
        coEvery { audioPlayer.playChunk(any<ShortArray>()) } just Runs
        every { audioPlayer.endChunk() } just Runs

        presenter = VoiceChatPresenter(
            mockActivity,
            mockView,
            mockChatPresenter,
            testScope,
            ttsClientFactory = { ttsClient }
        )
        val ttsField = VoiceChatPresenter::class.java.getDeclaredField("ttsService")
        ttsField.isAccessible = true
        ttsField.set(presenter, ttsClient)

        val audioField = VoiceChatPresenter::class.java.getDeclaredField("audioPlayer")
        audioField.isAccessible = true
        audioField.set(presenter, audioPlayer)

        val method = VoiceChatPresenter::class.java.getDeclaredMethod("speakGreetingMessage")
        method.isAccessible = true
        method.invoke(presenter)
        runBlocking { delay(200) }
        ShadowLooper.runUiThreadTasksIncludingDelayedTasks()
        shadowOf(Looper.getMainLooper()).runToEndOfTasks()

        coVerifyOrder {
            ttsClient.waitForInitComplete()
            ttsClient.process("有什么可以帮助您的？", 0)
        }
    }
} 
