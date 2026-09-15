package com.alibaba.mnnllm.android.agent

import android.content.Context
import android.util.Log
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.Serializable
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import java.io.File

object AgenticPythonEngine {
    private const val TAG = "AgenticPythonEngine"
    private const val MAX_CODE_CHARS = 12_000
    private const val MAX_INPUT_CHARS = 24_000
    private const val MAX_OUTPUT_CHARS = 24_000
    private const val DEFAULT_TIMEOUT_MS = 15_000L
    private const val MAX_TIMEOUT_MS = 30_000L

    private val json = Json { ignoreUnknownKeys = true }

    @Volatile
    private var initialized = false

    @Volatile
    private var appContext: Context? = null

    fun initialize(context: Context) {
        appContext = context.applicationContext
    }

    fun workspaceDir(context: Context? = appContext): File {
        val base = requireNotNull(context?.applicationContext ?: appContext) {
            "Python workspace requires initialized context."
        }
        return File(base.filesDir, "agent_workspace").apply { mkdirs() }
    }

    private fun ensureStarted(): Boolean {
        if (initialized) return true
        synchronized(this) {
            if (initialized) return true
            val context = appContext ?: return false
            if (!Python.isStarted()) {
                Python.start(AndroidPlatform(context))
            }
            initialized = true
            Log.i(TAG, "python runtime initialized")
        }
        return true
    }

    suspend fun execute(code: String, input: String = "", timeoutMs: Long = DEFAULT_TIMEOUT_MS): AgenticPythonResult {
        val trimmedCode = code.take(MAX_CODE_CHARS)
        val trimmedInput = input.take(MAX_INPUT_CHARS)
        val boundedTimeout = timeoutMs.coerceIn(1_000L, MAX_TIMEOUT_MS)
        return withContext(Dispatchers.Default) {
            if (!ensureStarted()) {
                return@withContext AgenticPythonResult(
                    ok = false,
                    error = "Python runtime is not initialized."
                )
            }
            runCatching {
                val module = Python.getInstance().getModule("agent_python")
                val raw = module.callAttr(
                    "run_code",
                    trimmedCode,
                    trimmedInput,
                    boundedTimeout,
                    workspaceDir().absolutePath
                ).toString()
                json.decodeFromString<AgenticPythonResult>(raw)
            }.getOrElse { error ->
                Log.e(TAG, "python_exec failed: ${error.message}", error)
                AgenticPythonResult(ok = false, error = error.message ?: error::class.java.simpleName)
            }.trimmed()
        }
    }

    private fun AgenticPythonResult.trimmed(): AgenticPythonResult {
        return copy(
            stdout = stdout.take(MAX_OUTPUT_CHARS),
            stderr = stderr.take(MAX_OUTPUT_CHARS),
            error = error.take(MAX_OUTPUT_CHARS)
        )
    }
}

@Serializable
data class AgenticPythonResult(
    val ok: Boolean = false,
    val stdout: String = "",
    val stderr: String = "",
    val error: String = "",
    val result: JsonElement? = null,
    val files: List<AgenticPythonFile> = emptyList(),
    val elapsed_ms: Int = 0
)

@Serializable
data class AgenticPythonFile(
    val name: String = "",
    val path: String = "",
    val relative_path: String = "",
    val mime_type: String = "",
    val size_bytes: Long = 0L
)
