package com.alibaba.mnnllm.api.openai.debug

import com.alibaba.mnnllm.api.openai.runtime.EnsureSessionResult
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.ByteArrayOutputStream
import java.io.PrintStream

class LlmDumperPluginTest {
    private class FakeController : LlmDebugController {
        var hasSession = false
        var currentModelId: String? = null
        var currentThinkingEnabled: Boolean? = null
        var ensureResult = EnsureSessionResult(success = false, reason = "SESSION_INIT_FAILED")
        var releaseCalled = false
        var setThinkingCalled = false

        override fun ensureSession(modelId: String, forceReload: Boolean): EnsureSessionResult {
            this.currentModelId = modelId
            return ensureResult
        }

        override fun getModelId(): String? = currentModelId

        override fun getThinkingEnabled(): Boolean? = currentThinkingEnabled

        override fun setThinkingEnabled(enabled: Boolean): Boolean {
            setThinkingCalled = true
            currentThinkingEnabled = enabled
            return hasSession
        }

        override fun hasSession(): Boolean = hasSession

        override fun releaseSession() {
            releaseCalled = true
            hasSession = false
            currentModelId = null
            currentThinkingEnabled = null
        }
    }

    @Test
    fun `plugin name should be llm`() {
        val plugin = LlmDumperPlugin(FakeController())
        assertEquals("llm", plugin.name)
    }

    @Test
    fun `ensure should print failure without model id`() {
        val plugin = LlmDumperPlugin(FakeController())
        val out = ByteArrayOutputStream()

        plugin.execute(listOf("ensure"), PrintStream(out))

        val output = out.toString()
        assertTrue(output.contains("RESULT=FAIL"))
        assertTrue(output.contains("REASON=MISSING_MODEL_ID"))
    }

    @Test
    fun `ensure should print success when controller returns session`() {
        val controller = FakeController().apply {
            hasSession = true
            currentModelId = "ModelScope/MNN/Qwen3.5-0.8B-MNN"
            currentThinkingEnabled = true
            ensureResult = EnsureSessionResult(
                success = true,
                modelId = currentModelId
            )
        }
        val plugin = LlmDumperPlugin(controller)
        val out = ByteArrayOutputStream()

        plugin.execute(listOf("ensure", "ModelScope/MNN/Qwen3.5-0.8B-MNN"), PrintStream(out))

        val output = out.toString()
        assertTrue(output.contains("RESULT=OK"))
        assertTrue(output.contains("SESSION_SOURCE=service_runtime"))
        assertTrue(output.contains("MODEL_ID=ModelScope/MNN/Qwen3.5-0.8B-MNN"))
    }

    @Test
    fun `thinking get should fail when no active session`() {
        val plugin = LlmDumperPlugin(FakeController())
        val out = ByteArrayOutputStream()

        plugin.execute(listOf("thinking", "get"), PrintStream(out))

        val output = out.toString()
        assertTrue(output.contains("RESULT=FAIL"))
        assertTrue(output.contains("REASON=NO_ACTIVE_SESSION"))
    }

    @Test
    fun `thinking set should switch value when session exists`() {
        val controller = FakeController().apply {
            hasSession = true
            currentModelId = "ModelScope/MNN/Qwen3.5-0.8B-MNN"
            currentThinkingEnabled = false
        }
        val plugin = LlmDumperPlugin(controller)
        val out = ByteArrayOutputStream()

        plugin.execute(listOf("thinking", "set", "on"), PrintStream(out))

        val output = out.toString()
        assertTrue(controller.setThinkingCalled)
        assertTrue(output.contains("RESULT=OK"))
        assertTrue(output.contains("THINKING_ENABLED=true"))
    }

    @Test
    fun `release should clear session`() {
        val controller = FakeController().apply {
            hasSession = true
            currentModelId = "ModelScope/MNN/Qwen3.5-0.8B-MNN"
            currentThinkingEnabled = true
        }
        val plugin = LlmDumperPlugin(controller)
        val out = ByteArrayOutputStream()

        plugin.execute(listOf("release"), PrintStream(out))

        val output = out.toString()
        assertTrue(controller.releaseCalled)
        assertTrue(output.contains("RESULT=OK"))
        assertTrue(output.contains("SESSION_PRESENT=false"))
    }
}
