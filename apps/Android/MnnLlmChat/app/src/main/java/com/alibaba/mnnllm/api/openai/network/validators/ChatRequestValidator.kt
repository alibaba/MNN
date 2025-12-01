package com.alibaba.mnnllm.api.openai.network.validators

import OpenAIChatRequest

/** * request verification * responsible for verificationrequestdatavalidity*/
class ChatRequestValidator {

    /** * verificationchatrequest * @param request chat request object * @return verification result,containingwhether validanderrorinfo*/
    fun validateChatRequest(request: OpenAIChatRequest): ValidationResult {
        //checkmessagelistwhether empty
        if (request.messages.isNullOrEmpty()) {
            return ValidationResult(
                isValid = false,
                errorMessage = "Messages array is empty"
            )
        }

        //canaddmoreverificationrules
        //e.g.：checkmessageformat、modelnameetc.
        
        return ValidationResult(isValid = true)
    }

    /**
     * verificationresultdataclass*/
    data class ValidationResult(
        val isValid: Boolean,
        val errorMessage: String? = null
    )
}