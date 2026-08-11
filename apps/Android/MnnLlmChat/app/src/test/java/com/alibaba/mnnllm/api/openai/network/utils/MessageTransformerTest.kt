package com.alibaba.mnnllm.api.openai.network.utils

import org.junit.Assert.assertEquals
import org.junit.Test

class MessageTransformerTest {

    @Test
    fun buildApiSystemPrompt_ignoresRuntimeDefaultPromptWhenRequestHasNoSystemMessage() {
        assertEquals(
            "",
            buildApiSystemPrompt(
                sessionPrompt = "You are a helpful assistant.",
                rawSystemPrompt = "",
                isR1Model = false
            )
        )
    }

    @Test
    fun buildApiSystemPrompt_keepsRequestSpecificSystemMessage() {
        assertEquals(
            "Reply in JSON only.",
            buildApiSystemPrompt(
                sessionPrompt = "You are a helpful assistant.",
                rawSystemPrompt = "Reply in JSON only.",
                isR1Model = false
            )
        )
    }
}
