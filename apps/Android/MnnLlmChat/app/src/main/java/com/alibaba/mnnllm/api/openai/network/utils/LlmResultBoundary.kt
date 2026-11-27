package com.alibaba.mnnllm.api.openai.network.utils

import android.util.Pair
import com.alibaba.mnnllm.android.llm.GenerateProgressListener
import com.alibaba.mnnllm.android.llm.LlmSession
import kotlinx.coroutines.CancellationException
import com.alibaba.mnnllm.api.openai.di.ServiceLocator
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put

internal class LlmGenerationException(message: String) : IllegalStateException(message)

internal object LlmResultBoundary {
    fun submitFullHistory(
        session: LlmSession,
        history: List<Pair<String, String>>,
        listener: GenerateProgressListener
    ): HashMap<String, Any> {
        val result = try {
            session.submitFullHistory(history, listener)
        } catch (e: CancellationException) {
            throw e
        } catch (e: OutOfMemoryError) {
            ServiceLocator.getLlmRuntimeController().releaseSession(expectedSession = session)
            throw LlmGenerationException("Not enough memory to generate; reload the model")
        } catch (e: Exception) {
            if (!session.isModelLoaded()) {
                ServiceLocator.getLlmRuntimeController().releaseSession(expectedSession = session)
            }
            throw e
        }
        return requireSuccess(session, result)
    }

    fun requireSuccess(session: LlmSession, result: HashMap<String, Any>): HashMap<String, Any> {
        val invalid = result["session_invalid"] == true
        if (invalid) ServiceLocator.getLlmRuntimeController().releaseSession(expectedSession = session)
        if (result["error"] == true || invalid) {
            val message = (result["message"] as? String)?.lineSequence()?.firstOrNull()?.take(256)
            throw LlmGenerationException(message?.takeIf { it.isNotBlank() } ?: "Model generation failed")
        }
        return result
    }

    fun errorJson(error: Exception, anthropic: Boolean = false): String = buildJsonObject {
        if (anthropic) put("type", "error")
        put("error", buildJsonObject {
            put("type", if (anthropic) "api_error" else "server_error")
            put("message", if (error is LlmGenerationException) error.message ?: "Model generation failed" else "Model generation failed")
        })
    }.toString()
}
