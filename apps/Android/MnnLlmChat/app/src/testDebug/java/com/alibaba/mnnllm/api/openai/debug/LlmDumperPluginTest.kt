package com.alibaba.mnnllm.api.openai.debug

import com.alibaba.mnnllm.api.openai.runtime.EnsureSessionResult
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
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
        var runResult = LlmRunResult(success = false, reason = "SESSION_INIT_FAILED")
        var lastRunModelId: String? = null
        var lastRunPrompt: String? = null
        var lastForceReload = false
        var lastUseAppConfig = false

        override fun ensureSession(modelId: String, forceReload: Boolean, useAppConfig: Boolean): EnsureSessionResult {
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

        override fun runPrompt(modelId: String, prompt: String, forceReload: Boolean, useAppConfig: Boolean): LlmRunResult {
            currentModelId = modelId
            lastRunModelId = modelId
            lastRunPrompt = prompt
            lastForceReload = forceReload
            lastUseAppConfig = useAppConfig
            return runResult
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

    @Test
    fun `run should execute prompt through ensured session`() {
        val controller = FakeController().apply {
            runResult = LlmRunResult(
                success = true,
                modelId = "test-model",
                response = "hello world",
                stage = "completed",
                submitReturned = true,
                chunkCount = 2,
                completionReceived = true,
                terminalCallbackCount = 1,
                terminalSignal = "callback_null",
                nonTerminalChunkCount = 2,
                emptyChunkCount = 0
            )
        }
        val plugin = LlmDumperPlugin(controller)
        val out = ByteArrayOutputStream()

        plugin.execute(listOf("run", "test-model", "say", "hi"), PrintStream(out))

        val output = out.toString()
        assertTrue(output.contains("RESULT=OK"))
        assertTrue(output.contains("MODEL_ID=test-model"))
        assertTrue(output.contains("TERMINAL_CALLBACK_COUNT=1"))
        assertTrue(output.contains("TERMINAL_SIGNAL=callback_null"))
        assertTrue(output.contains("NON_TERMINAL_CHUNK_COUNT=2"))
        assertTrue(output.contains("EMPTY_CHUNK_COUNT=0"))
        assertTrue(output.contains("RESPONSE_BEGIN"))
        assertTrue(output.contains("hello world"))
        assertTrue(output.contains("RESPONSE_END"))
        assertEquals("test-model", controller.lastRunModelId)
        assertEquals("say hi", controller.lastRunPrompt)
    }

    @Test
    fun `run should fail when prompt is missing`() {
        val plugin = LlmDumperPlugin(FakeController())
        val out = ByteArrayOutputStream()

        plugin.execute(listOf("run", "test-model"), PrintStream(out))

        val output = out.toString()
        assertTrue(output.contains("RESULT=FAIL"))
        assertTrue(output.contains("REASON=MISSING_MODEL_ID_OR_PROMPT"))
        assertTrue(output.contains("USAGE=dumpapp llm run <modelId> <prompt> [--force-reload] [--use-app-config]"))
    }

    @Test
    fun `run with --use-app-config should pass flag to controller`() {
        val controller = FakeController().apply {
            runResult = LlmRunResult(
                success = true,
                modelId = "test-model",
                response = "ok",
                stage = "completed",
                submitReturned = true,
                chunkCount = 1,
                completionReceived = true,
                terminalCallbackCount = 1,
                terminalSignal = "callback_null",
                nonTerminalChunkCount = 1,
                emptyChunkCount = 0
            )
        }
        val plugin = LlmDumperPlugin(controller)
        val out = ByteArrayOutputStream()

        plugin.execute(listOf("run", "test-model", "hi", "--use-app-config"), PrintStream(out))

        val output = out.toString()
        assertTrue(output.contains("RESULT=OK"))
        assertEquals("test-model", controller.lastRunModelId)
        assertEquals("hi", controller.lastRunPrompt)
        assertTrue("runPrompt should be called with useAppConfig=true", controller.lastUseAppConfig)
    }

    @Test
    fun `run should surface ensure failure reason`() {
        val controller = FakeController().apply {
            runResult = LlmRunResult(
                success = false,
                modelId = "bad-model",
                reason = "MODEL_CONFIG_NOT_FOUND",
                stage = "ensure",
                terminalCallbackCount = 0,
                terminalSignal = "missing",
                nonTerminalChunkCount = 0,
                emptyChunkCount = 0
            )
        }
        val plugin = LlmDumperPlugin(controller)
        val out = ByteArrayOutputStream()

        plugin.execute(listOf("run", "bad-model", "hello"), PrintStream(out))

        val output = out.toString()
        assertTrue(output.contains("RESULT=FAIL"))
        assertTrue(output.contains("MODEL_ID=bad-model"))
        assertTrue(output.contains("REASON=MODEL_CONFIG_NOT_FOUND"))
        assertTrue(output.contains("STAGE=ensure"))
        assertTrue(output.contains("TERMINAL_CALLBACK_COUNT=0"))
        assertTrue(output.contains("TERMINAL_SIGNAL=missing"))
        assertTrue(output.contains("NON_TERMINAL_CHUNK_COUNT=0"))
    }

    @Test
    fun `completeRunResult keeps missing terminal callback visible when synchronous completion is inferred`() {
        val startedAt = System.currentTimeMillis() - 10

        val result = completeRunResult(
            modelId = "test-model",
            startedAt = startedAt,
            response = "done",
            submitReturned = true,
            chunkCount = 1,
            completionReceived = false,
            terminalCallbackCount = 0,
            nonTerminalChunkCount = 1,
            emptyChunkCount = 0
        )

        assertTrue(result.success)
        assertEquals("completed_without_terminal_callback", result.stage)
        assertTrue(result.completionReceived)
        assertEquals(0, result.terminalCallbackCount)
        assertEquals("missing", result.terminalSignal)
        assertEquals(1, result.nonTerminalChunkCount)
    }

    @Test
    fun `completeRunResult treats synchronous return as completion without terminal callback`() {
        val startedAt = System.currentTimeMillis() - 10

        val result = completeRunResult(
            modelId = "test-model",
            startedAt = startedAt,
            response = "done",
            submitReturned = true,
            chunkCount = 1,
            completionReceived = false,
            terminalCallbackCount = 0,
            nonTerminalChunkCount = 1,
            emptyChunkCount = 0
        )

        assertTrue(result.success)
        assertEquals("completed_without_terminal_callback", result.stage)
        assertTrue(result.completionReceived)
        assertFalse(result.elapsedMs < 0)
    }
}
