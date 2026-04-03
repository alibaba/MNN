package com.alibaba.mnnllm.api.openai.service

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ApiRuntimeSessionStartPolicyTest {

    @Test
    fun `same model with loaded active session should reuse runtime session`() {
        assertTrue(
            ApiRuntimeSessionStartPolicy.shouldReuseLoadedSession(
                requestedModelId = "ModelScope/MNN/Qwen3.5-0.8B-MNN",
                activeModelId = "ModelScope/MNN/Qwen3.5-0.8B-MNN",
                hasLoadedActiveSession = true
            )
        )
    }

    @Test
    fun `different model should not reuse runtime session`() {
        assertFalse(
            ApiRuntimeSessionStartPolicy.shouldReuseLoadedSession(
                requestedModelId = "ModelScope/MNN/Qwen3.5-2B-MNN",
                activeModelId = "ModelScope/MNN/Qwen3.5-0.8B-MNN",
                hasLoadedActiveSession = true
            )
        )
    }

    @Test
    fun `missing loaded session should not reuse runtime session`() {
        assertFalse(
            ApiRuntimeSessionStartPolicy.shouldReuseLoadedSession(
                requestedModelId = "ModelScope/MNN/Qwen3.5-0.8B-MNN",
                activeModelId = "ModelScope/MNN/Qwen3.5-0.8B-MNN",
                hasLoadedActiveSession = false
            )
        )
    }
}
