package com.alibaba.mnnllm.api.openai.debug

import com.alibaba.mnnllm.api.openai.di.ServiceLocator
import com.alibaba.mnnllm.api.openai.runtime.EnsureSessionResult
import com.facebook.stetho.dumpapp.DumperContext
import com.facebook.stetho.dumpapp.DumperPlugin
import java.io.PrintStream

internal interface LlmDebugController {
    fun ensureSession(modelId: String, forceReload: Boolean): EnsureSessionResult
    fun getModelId(): String?
    fun getThinkingEnabled(): Boolean?
    fun setThinkingEnabled(enabled: Boolean): Boolean
    fun hasSession(): Boolean
    fun releaseSession()
}

internal object DefaultLlmDebugController : LlmDebugController {
    private val runtime get() = ServiceLocator.getLlmRuntimeController()

    override fun ensureSession(modelId: String, forceReload: Boolean): EnsureSessionResult {
        return runtime.ensureSession(modelId, forceReload)
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
        writer.println("  thinking get                          Get runtime thinking mode")
        writer.println("  thinking set <on|off>                Set runtime thinking mode")
        writer.println("  release                              Release runtime session")
    }
}
