package com.alibaba.mnnllm.api.openai.providers

import com.alibaba.mnnllm.android.chat.ChatActivity
import com.alibaba.mnnllm.android.llm.LlmSession
import com.alibaba.mnnllm.api.openai.interfaces.ChatSessionProvider

/**
 * 默认的聊天会话提供者实现
 * 作为ChatPresenter的适配器，提供对聊天会话的访问
 * 
 * 这个实现类封装了对ChatPresenter的访问，
 * 避免api.openai模块直接依赖ChatPresenter的内部实现
 */
class ChatSessionProviderImpl : ChatSessionProvider {
    
    /**
     * 获取当前的LLM会话实例
     * 通过ChatActivity获取ChatPresenter，然后安全地访问chatSession
     */
    override fun getLlmSession(): LlmSession? {
        return try {
            val chatPresenter = ChatActivity.getChatPresenter()
            // 使用反射或者添加公共方法来访问chatSession
            // 这里需要ChatPresenter提供一个公共方法来获取LlmSession
            chatPresenter?.getLlmSession()
        } catch (e: Exception) {
            null
        }
    }
    
    /**
     * 检查是否有活跃的聊天会话
     */
    override fun hasActiveSession(): Boolean {
        return getLlmSession() != null
    }
    
    /**
     * 获取当前会话ID
     */
    override fun getCurrentSessionId(): String? {
        return try {
            val chatPresenter = ChatActivity.getChatPresenter()
            chatPresenter?.getSessionId()
        } catch (e: Exception) {
            null
        }
    }
}