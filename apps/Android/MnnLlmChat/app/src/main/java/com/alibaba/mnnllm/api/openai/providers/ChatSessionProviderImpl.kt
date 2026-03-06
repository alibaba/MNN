package com.alibaba.mnnllm.api.openai.providers

import com.alibaba.mnnllm.android.chat.ChatActivity
import com.alibaba.mnnllm.android.llm.LlmSession
import com.alibaba.mnnllm.api.openai.di.ServiceLocator
import com.alibaba.mnnllm.api.openai.interfaces.ChatSessionProvider

/** * defaultchatsessionproviderimplementation * asChatPresenteradapter,providetochatsessionaccessing * * thisimplementationclassencapsulatetoChatPresenteraccessing, * avoidapi.openaimoduledirectlydependencyChatPresenterinternalimplementation*/
class ChatSessionProviderImpl : ChatSessionProvider {
    
    /** * getcurrentLLM sessioninstance * throughChatActivitygetChatPresenter,thensafelyaccessingchatSession*/
    override fun getLlmSession(): LlmSession? {
        return try {
            val runtimeSession = ServiceLocator.getLlmRuntimeController().getActiveSession()
            if (runtimeSession != null) {
                return runtimeSession
            }

            val chatPresenter = ChatActivity.getChatPresenter()
            // use reflection or add public method to access chatSession
            // here need ChatPresenter provide a public method to get LlmSession
            chatPresenter?.getLlmSession()
        } catch (e: Exception) {
            null
        }
    }
    
    /** * checkwhether there isactivechatsession*/
    override fun hasActiveSession(): Boolean {
        return getLlmSession() != null
    }
    
    /** * getcurrentsessionID*/
    override fun getCurrentSessionId(): String? {
        return try {
            val runtimeSession = ServiceLocator.getLlmRuntimeController().getActiveSession()
            if (runtimeSession != null) {
                return runtimeSession.sessionId
            }
            val chatPresenter = ChatActivity.getChatPresenter()
            chatPresenter?.getSessionId()
        } catch (e: Exception) {
            null
        }
    }
}
