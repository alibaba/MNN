package com.alibaba.mnnllm.api.openai.di

import com.alibaba.mnnllm.api.openai.interfaces.ChatSessionProvider
import com.alibaba.mnnllm.api.openai.providers.ChatSessionProviderImpl

/**
 * 服务定位器
 * 用于管理api.openai模块的依赖注入
 * 
 * 这个类提供了一个简单的依赖注入机制，
 * 避免硬编码依赖，便于测试和模块化
 */
object ServiceLocator {
    
    private var _chatSessionProvider: ChatSessionProvider? = null
    
    /**
     * 获取聊天会话提供者
     * 如果没有设置，则返回默认实现
     */
    fun getChatSessionProvider(): ChatSessionProvider {
        return _chatSessionProvider ?: ChatSessionProviderImpl().also {
            _chatSessionProvider = it
        }
    }
    
    /**
     * 设置聊天会话提供者
     * 主要用于测试或者自定义实现
     */
    fun setChatSessionProvider(provider: ChatSessionProvider) {
        _chatSessionProvider = provider
    }
    
    /**
     * 重置服务定位器
     * 主要用于测试清理
     */
    fun reset() {
        _chatSessionProvider = null
    }
}