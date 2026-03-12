package com.alibaba.mnnllm.api.openai.service

import android.content.ComponentName
import android.content.Context
import io.mockk.every
import io.mockk.mockk
import io.mockk.verify
import org.junit.After
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [28])
class OpenAIServiceLifecycleTest {

    @After
    fun tearDown() {
    }

    @Test
    fun startService_doesNotBindService() {
        val context = mockk<Context>(relaxed = true)
        every { context.startForegroundService(any()) } returns mockk<ComponentName>(relaxed = true)

        OpenAIService.startService(context, "ModelScope/MNN/Qwen3.5-0.8B-MNN")

        verify(exactly = 1) { context.startForegroundService(any()) }
        verify(exactly = 0) {
            context.bindService(any<android.content.Intent>(), any(), any<Int>())
        }
    }
}
