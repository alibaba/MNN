package com.alibaba.mnnllm.api.openai.debug

import android.util.Pair
import com.alibaba.mnnllm.android.llm.GenerateProgressListener
import com.alibaba.mnnllm.api.openai.di.ServiceLocator
import com.alibaba.mnnllm.api.openai.runtime.EnsureSessionResult
import com.facebook.stetho.dumpapp.DumperContext
import com.facebook.stetho.dumpapp.DumperPlugin
import java.io.PrintStream
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

internal data class LlmRunResult(
    val success: Boolean,
    val modelId: String? = null,
    val reason: String? = null,
    val response: String = "",
    val stage: String = "unknown",
    val submitReturned: Boolean = false,
    val chunkCount: Int = 0,
    val completionReceived: Boolean = false,
    val elapsedMs: Long = 0L
)

internal interface LlmDebugController {
    fun ensureSession(modelId: String, forceReload: Boolean, useAppConfig: Boolean = false): EnsureSessionResult
    fun getModelId(): String?
    fun getThinkingEnabled(): Boolean?
    fun setThinkingEnabled(enabled: Boolean): Boolean
    fun hasSession(): Boolean
    fun releaseSession()
    fun runPrompt(modelId: String, prompt: String, forceReload: Boolean, useAppConfig: Boolean = false): LlmRunResult
}

internal object DefaultLlmDebugController : LlmDebugController {
    private val runtime get() = ServiceLocator.getLlmRuntimeController()

    override fun ensureSession(modelId: String, forceReload: Boolean, useAppConfig: Boolean): EnsureSessionResult {
        return runtime.ensureSession(modelId, forceReload, useAppConfig)
    }

    override fun getModelId(): String? = runtime.getActiveModelId()

    override fun getThinkingEnabled(): Boolean? = runtime.getThinkingEnabled()

    override fun setThinkingEnabled(enabled: Boolean): Boolean {
        return runtime.setThinkingEnabled(enabled)
    }

    override fun hasSession(): Boolean = runtime.getActiveSession() != null

    override fun releaseSession() {
        runtime.releaseSession()
    }

    override fun runPrompt(modelId: String, prompt: String, forceReload: Boolean, useAppConfig: Boolean): LlmRunResult {
        val startedAt = System.currentTimeMillis()
        val ensureResult = if (useAppConfig && !forceReload && runtime.getActiveSession() != null && runtime.getActiveModelId() == modelId) {
            EnsureSessionResult(success = true, session = runtime.getActiveSession(), modelId = modelId)
        } else {
            runtime.ensureSession(modelId, forceReload, useAppConfig)
        }
        val session = ensureResult.session ?: runtime.getActiveSession()
        if (!ensureResult.success || session == null) {
            return LlmRunResult(
                success = false,
                modelId = ensureResult.modelId ?: modelId,
                reason = ensureResult.reason ?: "SESSION_NOT_READY",
                stage = "ensure",
                elapsedMs = System.currentTimeMillis() - startedAt
            )
        }

        val responseBuilder = StringBuilder()
        val latch = CountDownLatch(1)
        var chunkCount = 0
        var completionReceived = false
        var submitReturned = false

        val listener = object : GenerateProgressListener {
            override fun onProgress(progress: String?): Boolean {
                if (progress != null) {
                    chunkCount += 1
                    responseBuilder.append(progress)
                } else {
                    completionReceived = true
                    latch.countDown()
                }
                return false
            }
        }
        return runCatching {
            if (useAppConfig) {
                session.generate(prompt, emptyMap(), listener)
            } else {
                session.submitFullHistory(listOf(Pair("user", prompt)), listener)
            }
            submitReturned = true

            val completed = latch.await(60, TimeUnit.SECONDS)
            if (!completed) {
                return LlmRunResult(
                    success = false,
                    modelId = ensureResult.modelId ?: modelId,
                    reason = "RUN_TIMEOUT",
                    response = responseBuilder.toString(),
                    stage = "await_completion",
                    submitReturned = submitReturned,
                    chunkCount = chunkCount,
                    completionReceived = completionReceived,
                    elapsedMs = System.currentTimeMillis() - startedAt
                )
            }

            LlmRunResult(
                success = true,
                modelId = ensureResult.modelId ?: modelId,
                response = responseBuilder.toString(),
                stage = "completed",
                submitReturned = submitReturned,
                chunkCount = chunkCount,
                completionReceived = completionReceived,
                elapsedMs = System.currentTimeMillis() - startedAt
            )
        }.getOrElse {
            LlmRunResult(
                success = false,
                modelId = ensureResult.modelId ?: modelId,
                reason = "RUN_FAILED",
                response = responseBuilder.toString(),
                stage = if (submitReturned) "after_submit_exception" else "submit",
                submitReturned = submitReturned,
                chunkCount = chunkCount,
                completionReceived = completionReceived,
                elapsedMs = System.currentTimeMillis() - startedAt
            )
        }
    }
}

internal class LlmDumperPlugin(
    private val controller: LlmDebugController = DefaultLlmDebugController
) : DumperPlugin {
    override fun getName(): String = "llm"

    override fun dump(dumpContext: DumperContext) {
        execute(dumpContext.argsAsList, dumpContext.stdout)
    }

    internal fun execute(args: List<String>, writer: PrintStream) {
        if (args.isEmpty()) {
            printUsage(writer)
            return
        }

        when (args[0]) {
            "status" -> handleStatus(writer)
            "ensure" -> handleEnsure(writer, args.drop(1))
            "run" -> handleRun(writer, args.drop(1))
            "thinking" -> handleThinking(writer, args.drop(1))
            "release" -> handleRelease(writer)
            else -> printUsage(writer)
        }
    }

    private fun handleStatus(writer: PrintStream) {
        val hasSession = controller.hasSession()
        val modelId = controller.getModelId().orEmpty()
        val thinking = controller.getThinkingEnabled()
        writer.println("RESULT=OK")
        writer.println("SESSION_PRESENT=$hasSession")
        writer.println("SESSION_SOURCE=${if (hasSession) "service_runtime" else "none"}")
        writer.println("MODEL_ID=$modelId")
        writer.println("THINKING_ENABLED=${thinking ?: "unknown"}")
    }

    private fun handleEnsure(writer: PrintStream, args: List<String>) {
        if (args.isEmpty()) {
            writer.println("RESULT=FAIL")
            writer.println("REASON=MISSING_MODEL_ID")
            writer.println("USAGE=dumpapp llm ensure <modelId> [--force-reload]")
            return
        }
        val modelId = args[0]
        val forceReload = args.drop(1).any { it == "--force-reload" }
        val result = controller.ensureSession(modelId, forceReload)
        val hasSession = controller.hasSession()
        if (!result.success || !hasSession) {
            writer.println("RESULT=FAIL")
            writer.println("SESSION_PRESENT=$hasSession")
            writer.println("SESSION_SOURCE=${if (hasSession) "service_runtime" else "none"}")
            writer.println("MODEL_ID=${result.modelId ?: modelId}")
            writer.println("REASON=${result.reason ?: "SESSION_NOT_READY"}")
            return
        }
        writer.println("RESULT=OK")
        writer.println("SESSION_PRESENT=$hasSession")
        writer.println("SESSION_SOURCE=service_runtime")
        writer.println("MODEL_ID=${result.modelId ?: modelId}")
        writer.println("THINKING_ENABLED=${controller.getThinkingEnabled() ?: "unknown"}")
    }

    private fun handleRun(writer: PrintStream, args: List<String>) {
        val forceReload = args.any { it == "--force-reload" }
        val useAppConfig = args.any { it == "--use-app-config" }
        val positionalArgs = args.filterNot { it == "--force-reload" || it == "--use-app-config" }
        if (positionalArgs.size < 2) {
            writer.println("RESULT=FAIL")
            writer.println("REASON=MISSING_MODEL_ID_OR_PROMPT")
            writer.println("USAGE=dumpapp llm run <modelId> <prompt> [--force-reload] [--use-app-config]")
            return
        }

        val modelId = positionalArgs[0]
        val prompt = positionalArgs.drop(1).joinToString(" ")
        val result = controller.runPrompt(modelId, prompt, forceReload, useAppConfig)
        if (!result.success) {
            writer.println("RESULT=FAIL")
            writer.println("MODEL_ID=${result.modelId ?: modelId}")
            writer.println("REASON=${result.reason ?: "RUN_FAILED"}")
            writer.println("STAGE=${result.stage}")
            writer.println("SUBMIT_RETURNED=${result.submitReturned}")
            writer.println("CHUNK_COUNT=${result.chunkCount}")
            writer.println("COMPLETION_RECEIVED=${result.completionReceived}")
            writer.println("ELAPSED_MS=${result.elapsedMs}")
            writer.println("PARTIAL_RESPONSE_LEN=${result.response.length}")
            return
        }

        writer.println("RESULT=OK")
        writer.println("SESSION_PRESENT=true")
        writer.println("SESSION_SOURCE=service_runtime")
        writer.println("MODEL_ID=${result.modelId ?: modelId}")
        writer.println("STAGE=${result.stage}")
        writer.println("SUBMIT_RETURNED=${result.submitReturned}")
        writer.println("CHUNK_COUNT=${result.chunkCount}")
        writer.println("COMPLETION_RECEIVED=${result.completionReceived}")
        writer.println("ELAPSED_MS=${result.elapsedMs}")
        writer.println("RESPONSE_BEGIN")
        writer.println(result.response)
        writer.println("RESPONSE_END")
    }

    private fun handleThinking(writer: PrintStream, args: List<String>) {
        if (args.isEmpty()) {
            writer.println("RESULT=FAIL")
            writer.println("REASON=MISSING_THINKING_COMMAND")
            writer.println("USAGE=dumpapp llm thinking get|set <on|off>")
            return
        }

        when (args[0]) {
            "get" -> handleThinkingGet(writer)
            "set" -> handleThinkingSet(writer, args.drop(1))
            else -> {
                writer.println("RESULT=FAIL")
                writer.println("REASON=UNKNOWN_THINKING_COMMAND")
                writer.println("USAGE=dumpapp llm thinking get|set <on|off>")
            }
        }
    }

    private fun handleThinkingGet(writer: PrintStream) {
        if (!controller.hasSession()) {
            writer.println("RESULT=FAIL")
            writer.println("REASON=NO_ACTIVE_SESSION")
            writer.println("SESSION_PRESENT=false")
            return
        }
        val modelId = controller.getModelId().orEmpty()
        val thinking = controller.getThinkingEnabled()
        if (thinking == null) {
            writer.println("RESULT=FAIL")
            writer.println("REASON=THINKING_STATE_UNAVAILABLE")
            writer.println("SESSION_PRESENT=true")
            writer.println("MODEL_ID=$modelId")
            return
        }
        writer.println("RESULT=OK")
        writer.println("SESSION_PRESENT=true")
        writer.println("MODEL_ID=$modelId")
        writer.println("THINKING_ENABLED=$thinking")
    }

    private fun handleThinkingSet(writer: PrintStream, args: List<String>) {
        if (args.isEmpty()) {
            writer.println("RESULT=FAIL")
            writer.println("REASON=MISSING_THINKING_VALUE")
            writer.println("USAGE=dumpapp llm thinking set <on|off>")
            return
        }
        if (!controller.hasSession()) {
            writer.println("RESULT=FAIL")
            writer.println("REASON=NO_ACTIVE_SESSION")
            writer.println("SESSION_PRESENT=false")
            return
        }

        val value = args[0].lowercase()
        val enabled = when (value) {
            "on", "true", "1" -> true
            "off", "false", "0" -> false
            else -> {
                writer.println("RESULT=FAIL")
                writer.println("REASON=INVALID_THINKING_VALUE")
                writer.println("USAGE=dumpapp llm thinking set <on|off>")
                return
            }
        }

        val setOk = controller.setThinkingEnabled(enabled)
        if (!setOk) {
            writer.println("RESULT=FAIL")
            writer.println("REASON=SET_THINKING_FAILED")
            return
        }

        val thinking = controller.getThinkingEnabled()
        val modelId = controller.getModelId().orEmpty()
        writer.println("RESULT=OK")
        writer.println("SESSION_PRESENT=true")
        writer.println("MODEL_ID=$modelId")
        writer.println("THINKING_ENABLED=${thinking ?: "unknown"}")
    }

    private fun handleRelease(writer: PrintStream) {
        controller.releaseSession()
        writer.println("RESULT=OK")
        writer.println("SESSION_PRESENT=false")
        writer.println("SESSION_SOURCE=none")
    }

    private fun printUsage(writer: PrintStream) {
        writer.println("Usage: dumpapp llm <command>")
        writer.println("Commands:")
        writer.println("  status                               Show service-runtime session status")
        writer.println("  ensure <modelId> [--force-reload]    Ensure runtime session for model")
        writer.println("  run <modelId> <prompt> [--force-reload] [--use-app-config]  Run prompt via LlmSession")
        writer.println("  thinking get                          Get runtime thinking mode")
        writer.println("  thinking set <on|off>                Set runtime thinking mode")
        writer.println("  release                              Release runtime session")
    }
}
