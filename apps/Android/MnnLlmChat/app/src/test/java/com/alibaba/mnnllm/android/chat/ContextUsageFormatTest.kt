package com.alibaba.mnnllm.android.chat

import com.alibaba.mnnllm.android.modelsettings.ModelConfig
import org.junit.Assert.assertEquals
import org.junit.Test

class ContextUsageFormatTest {
    @Test
    fun `token counts are compact`() {
        assertEquals("0", ContextUsageFormat.tokens(-3))
        assertEquals("950", ContextUsageFormat.tokens(950))
        assertEquals("1.2K", ContextUsageFormat.tokens(1234))
        assertEquals("4K", ContextUsageFormat.tokens(4096))
        assertEquals("32K", ContextUsageFormat.tokens(32768))
        assertEquals("117K", ContextUsageFormat.tokens(120000))
    }

    @Test
    fun `context limit choices respect the model window and keep the current value`() {
        assertEquals(listOf(1024, 2048, 4096, 8192), ModelConfig.contextLimitChoices(8192, 4096))
        assertEquals(ModelConfig.CONTEXT_TOKEN_LIMIT_CHOICES, ModelConfig.contextLimitChoices(0, 4096))
        assertEquals(listOf(1024, 2048, 3000), ModelConfig.contextLimitChoices(3000, 3000))
        assertEquals(listOf(512), ModelConfig.contextLimitChoices(512, 512))
        assertEquals(listOf(4096), ModelConfig.contextLimitChoices(512, 0))
    }
}
