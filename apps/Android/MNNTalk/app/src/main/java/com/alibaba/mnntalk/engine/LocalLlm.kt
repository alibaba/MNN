package com.alibaba.mnntalk.engine

import java.io.Closeable
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Small JNI surface for the demo. It intentionally exposes only the operations
 * needed by a local, streaming chat session.
 */
class LocalLlm private constructor(private var nativeHandle: Long) : Closeable {
    fun interface TokenListener {
        /**
         * @return true to stop generation after this token.
         */
        fun onToken(token: String): Boolean
    }

    private val closed = AtomicBoolean(false)

    fun generate(prompt: String, listener: TokenListener): GenerationMetrics {
        check(!closed.get() && nativeHandle != 0L) { "Local LLM is closed" }
        val values = nativeGenerate(nativeHandle, prompt, listener)
        return GenerationMetrics(
            promptTokens = values.getOrElse(0) { 0L },
            generatedTokens = values.getOrElse(1) { 0L },
            prefillMicros = values.getOrElse(2) { 0L },
            decodeMicros = values.getOrElse(3) { 0L }
        )
    }

    fun stop() {
        if (!closed.get() && nativeHandle != 0L) {
            nativeStop(nativeHandle)
        }
    }

    fun reset() {
        if (!closed.get() && nativeHandle != 0L) {
            nativeReset(nativeHandle)
        }
    }

    override fun close() {
        if (closed.compareAndSet(false, true)) {
            val handle = nativeHandle
            nativeHandle = 0L
            if (handle != 0L) {
                nativeStop(handle)
                nativeRelease(handle)
            }
        }
    }

    private external fun nativeGenerate(
        handle: Long,
        prompt: String,
        listener: TokenListener
    ): LongArray

    private external fun nativeStop(handle: Long)
    private external fun nativeReset(handle: Long)
    private external fun nativeRelease(handle: Long)

    companion object {
        init {
            System.loadLibrary("mnnvoice")
        }

        fun load(
            configFile: File,
            cacheDirectory: File,
            systemPrompt: String,
            maxNewTokens: Int = 256
        ): LocalLlm {
            require(configFile.isFile) { "LLM config not found: ${configFile.absolutePath}" }
            cacheDirectory.mkdirs()
            val handle = nativeCreate(
                configFile.absolutePath,
                cacheDirectory.absolutePath,
                systemPrompt,
                maxNewTokens
            )
            check(handle != 0L) { "Failed to load local LLM from ${configFile.absolutePath}" }
            return LocalLlm(handle)
        }

        @JvmStatic
        private external fun nativeCreate(
            configPath: String,
            cachePath: String,
            systemPrompt: String,
            maxNewTokens: Int
        ): Long
    }
}
