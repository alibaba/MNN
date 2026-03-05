package com.alibaba.mnnllm.android.utils

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class CrashReportContextTest {

    @Test
    fun `buildLlmConfigLogPayload uses unknown stage and empty json fallback`() {
        val payload = CrashReportContext.buildLlmConfigLogPayload(" ", "", maxConfigChars = 64)
        assertTrue(payload.contains("[llm_set_config][unknown]"))
        assertTrue(payload.contains("len=2"))
        assertTrue(payload.endsWith("{}"))
    }

    @Test
    fun `buildLlmConfigLogPayload truncates long config and keeps source length`() {
        val source = "x".repeat(20)
        val payload = CrashReportContext.buildLlmConfigLogPayload("init_load", source, maxConfigChars = 8)
        assertTrue(payload.contains("[llm_set_config][init_load]"))
        assertTrue(payload.contains("len=20"))
        assertTrue(payload.contains("xxxxxxxx...(truncated, original_len=20)"))
    }

    @Test
    fun `buildLlmConfigLogPayload keeps full config when below threshold`() {
        val source = "{\"topK\":20}"
        val payload = CrashReportContext.buildLlmConfigLogPayload("update_config", source, maxConfigChars = 64)
        assertEquals("[llm_set_config][update_config] len=11 {\"topK\":20}", payload)
    }
}
