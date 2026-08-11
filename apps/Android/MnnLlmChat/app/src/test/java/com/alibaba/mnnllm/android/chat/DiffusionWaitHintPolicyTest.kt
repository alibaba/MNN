package com.alibaba.mnnllm.android.chat

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class DiffusionWaitHintPolicyTest {

    @Test
    fun `returns true for stable diffusion model`() {
        assertTrue(DiffusionWaitHintPolicy.shouldShowWaitHint("stable-diffusion-v1-5-mnn-opencl"))
    }

    @Test
    fun `returns true for sana model`() {
        assertTrue(DiffusionWaitHintPolicy.shouldShowWaitHint("sana-1600m"))
    }

    @Test
    fun `returns false for llm model`() {
        assertFalse(DiffusionWaitHintPolicy.shouldShowWaitHint("Qwen3.5-4B-MNN"))
    }
}
