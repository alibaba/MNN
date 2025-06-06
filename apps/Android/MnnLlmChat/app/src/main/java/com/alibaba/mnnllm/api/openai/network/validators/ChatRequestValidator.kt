package com.alibaba.mnnllm.api.openai.network.validators

import OpenAIChatRequest

/**
 * 请求验证器
 * 负责验证请求数据的有效性
 */
class ChatRequestValidator {

    /**
     * 验证聊天请求
     * @param request 聊天请求对象
     * @return 验证结果，包含是否有效和错误信息
     */
    fun validateChatRequest(request: OpenAIChatRequest): ValidationResult {
        // 检查消息列表是否为空
        if (request.messages.isNullOrEmpty()) {
            return ValidationResult(
                isValid = false,
                errorMessage = "Messages array is empty"
            )
        }

        // 可以添加更多验证规则
        // 例如：检查消息格式、模型名称等
        
        return ValidationResult(isValid = true)
    }

    /**
     * 验证结果数据类
     */
    data class ValidationResult(
        val isValid: Boolean,
        val errorMessage: String? = null
    )
}