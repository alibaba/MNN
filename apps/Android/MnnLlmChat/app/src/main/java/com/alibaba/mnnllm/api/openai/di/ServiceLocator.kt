package com.alibaba.mnnllm.api.openai.di

import com.alibaba.mnnllm.api.openai.interfaces.ChatSessionProvider
import com.alibaba.mnnllm.api.openai.providers.ChatSessionProviderImpl
import com.alibaba.mnnllm.api.openai.runtime.DefaultLlmRuntimeController
import com.alibaba.mnnllm.api.openai.runtime.LlmRuntimeController

/** * Service Locator * for managingapi.openaimoduledependencyinjection * * thisclassprovide asimpledependencyinjectionmechanism, * avoidhard-codeddependency,fortestandmodularization*/
object ServiceLocator {
    
    private var _chatSessionProvider: ChatSessionProvider? = null
    private var _llmRuntimeController: LlmRuntimeController? = null
    
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

    fun getLlmRuntimeController(): LlmRuntimeController {
        return _llmRuntimeController ?: DefaultLlmRuntimeController.also {
            _llmRuntimeController = it
        }
    }

    fun setLlmRuntimeController(controller: LlmRuntimeController) {
        _llmRuntimeController = controller
    }
    
    /** * resetService Locator * mainlyfortestcleanup*/
    fun reset() {
        _chatSessionProvider = null
        _llmRuntimeController = null
    }
}
