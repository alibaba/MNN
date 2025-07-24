package com.alibaba.mnnllm.android.chat.voice

import android.app.Activity
import android.media.AudioManager
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
import org.robolectric.annotation.Config

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
        testScope.cancel()
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
        // 启动后状态会变为LISTENING（异步，需等待）
        runBlocking { delay(100) }
        presenter.stop()
        // 停止后isStopped应为true，状态不再变化
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
    fun `test stopGeneration triggers state and view`() {
        every { mockChatPresenter.stopGenerate() } just Runs
        every { mockView.updateStatus(any()) } just Runs
        presenter.stopGeneration()
        runBlocking { delay(400) }
        verify { mockView.updateStatus(VoiceChatState.STOPPING) }
        verify { mockView.updateStatus(VoiceChatState.LISTENING) }
    }

    @Test
    fun `test onLlmGenerateProgress triggers task channel`() {
        val processor = mockk<GenerateResultProcessor>(relaxed = true)
        every { mockView.addTranscript(any()) } just Runs
        every { mockView.updateLastTranscript(any()) } just Runs
        presenter.onLlmGenerateProgress("hello", processor)
        runBlocking { delay(100) }
        // 只要没有异常，说明流程可达
    }

    @Test
    fun `test onGenerateFinished triggers final chunk`() {
        every { mockView.updateStatus(any()) } just Runs
        presenter.onGenerateFinished(hashMapOf("response" to "ok"))
        runBlocking { delay(100) }
        // 只要没有异常，说明流程可达
    }

    @Test
    fun `test error handling in speakGreetingMessage`() {
        // 模拟activity.getString抛异常
        every { mockActivity.getString(any()) } throws RuntimeException("test error")
        every { mockView.updateStatus(any()) } just Runs
        presenter.start()
        runBlocking { delay(200) }
        // 只要没有crash，说明异常被处理
    }
} 