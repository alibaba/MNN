package com.alibaba.mnnllm.api.openai.di

import com.alibaba.mnnllm.api.openai.interfaces.ChatSessionProvider
import com.alibaba.mnnllm.api.openai.providers.ChatSessionProviderImpl

/** * Service Locator * for managingapi.openaimoduledependencyinjection * * thisclassprovide asimpledependencyinjectionmechanism, * avoidhard-codeddependency,fortestandmodularization*/
object ServiceLocator {
    
    private var _chatSessionProvider: ChatSessionProvider? = null
    
    /** * Get chat session provider * if not availableset，thenreturndefaultimplementation*/
    fun getChatSessionProvider(): ChatSessionProvider {
        return _chatSessionProvider ?: ChatSessionProviderImpl().also {
            _chatSessionProvider = it
        }
    }
    
    /** * Set chat session provider * mainlyfortestorcustomimplementation*/
    fun setChatSessionProvider(provider: ChatSessionProvider) {
        _chatSessionProvider = provider
    }
    
    /** * resetService Locator * mainlyfortestcleanup*/
    fun reset() {
        _chatSessionProvider = null
    }
}