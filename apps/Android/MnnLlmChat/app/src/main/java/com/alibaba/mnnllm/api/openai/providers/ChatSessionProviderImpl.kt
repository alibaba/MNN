package com.alibaba.mnnllm.api.openai.providers

import com.alibaba.mnnllm.android.llm.LlmSession
import com.alibaba.mnnllm.api.openai.di.ServiceLocator
import com.alibaba.mnnllm.api.openai.interfaces.ChatSessionProvider

/**
 * Default ChatSessionProvider implementation.
 * Returns the active LlmSession from LlmRuntimeController (unified session management).
 * Chat and API both use the same session source.
 */
class ChatSessionProviderImpl : ChatSessionProvider {

    override fun getLlmSession(): LlmSession? {
        return try {
            ServiceLocator.getLlmRuntimeController().getActiveSession()
        } catch (e: Exception) {
            null
        }
    }

    override fun hasActiveSession(): Boolean {
        return getLlmSession() != null
    }

    override fun getCurrentSessionId(): String? {
        return try {
            ServiceLocator.getLlmRuntimeController().getActiveSession()?.sessionId
        } catch (e: Exception) {
            null
        }
    }
}
