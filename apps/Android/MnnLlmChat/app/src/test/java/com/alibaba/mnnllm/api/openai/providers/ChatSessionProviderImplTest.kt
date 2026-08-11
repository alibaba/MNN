package com.alibaba.mnnllm.api.openai.providers

import com.alibaba.mnnllm.api.openai.di.ServiceLocator
import com.alibaba.mnnllm.api.openai.runtime.LlmRuntimeController
import org.junit.After
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Before
import org.junit.Test
import org.mockito.Mockito.`when`
import org.mockito.Mockito.mock
import org.mockito.Mockito.reset

/**
 * Unit tests for ChatSessionProviderImpl (unified session via LlmRuntimeController).
 * Tests that use non-null LlmSession are omitted: LlmSession loads native libs and cannot be
 * mocked in JVM unit tests. The "session exists" path is covered by instrumented/UI tests.
 */
class ChatSessionProviderImplTest {

    private lateinit var mockController: LlmRuntimeController

    @Before
    fun setup() {
        mockController = mock(LlmRuntimeController::class.java)
        ServiceLocator.setLlmRuntimeController(mockController)
    }

    @After
    fun teardown() {
        ServiceLocator.reset()
        reset(mockController)
    }

    @Test
    fun `getLlmSession returns null when runtime has no active session`() {
        `when`(mockController.getActiveSession()).thenReturn(null)

        val provider = ServiceLocator.getChatSessionProvider()
        val session = provider.getLlmSession()

        assertNull(session)
    }

    @Test
    fun `hasActiveSession returns false when no session`() {
        `when`(mockController.getActiveSession()).thenReturn(null)

        val provider = ServiceLocator.getChatSessionProvider()

        assertFalse(provider.hasActiveSession())
    }

    @Test
    fun `getCurrentSessionId returns null when no session`() {
        `when`(mockController.getActiveSession()).thenReturn(null)

        val provider = ServiceLocator.getChatSessionProvider()

        assertNull(provider.getCurrentSessionId())
    }

    @Test
    fun `getLlmSession returns null when runtime throws`() {
        `when`(mockController.getActiveSession()).thenThrow(RuntimeException("test error"))

        val provider = ServiceLocator.getChatSessionProvider()
        val session = provider.getLlmSession()

        assertNull(session)
    }
}
