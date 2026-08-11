package com.alibaba.mnnllm.android.chat

import org.junit.Assert.assertEquals
import org.junit.Test

class ChatPresenterPromptTest {

    @Test
    fun `sana blank input should use sana default prompt`() {
        val resolved = ChatPresenter.resolveDiffusionPrompt("", "local/sana-1600m")
        assertEquals(DEFAULT_SANA_PROMPT, resolved)
    }

    @Test
    fun `sana non blank input should keep user input`() {
        val resolved = ChatPresenter.resolveDiffusionPrompt("custom text", "local/sana-1600m")
        assertEquals("custom text", resolved)
    }

    @Test
    fun `non sana blank input should keep generic diffusion default`() {
        val resolved = ChatPresenter.resolveDiffusionPrompt("", "qwen2.5-0.5b")
        assertEquals("A cyberpunk cat in neon lights", resolved)
    }
}
