package com.alibaba.mnnllm.api.openai.interfaces

import com.alibaba.mnnllm.android.llm.LlmSession

/**
 * 聊天会话提供者接口
 * 用于解耦api.openai模块与android.chat模块的直接依赖
 * 
 * 这个接口定义了api.openai模块需要的核心功能，
 * 避免直接访问ChatPresenter的私有成员
 */
interface ChatSessionProvider {
    
    /**
     * 获取当前的LLM会话实例
     * @return LlmSession实例，如果没有活跃会话则返回null
     */
    fun getLlmSession(): LlmSession?
    
    /**
     * 检查是否有活跃的聊天会话
     * @return true如果有活跃会话，false否则
     */
    fun hasActiveSession(): Boolean
    
    /**
     * 获取当前会话ID
     * @return 会话ID，如果没有活跃会话则返回null
     */
    fun getCurrentSessionId(): String?
}