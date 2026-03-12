package com.alibaba.mnnllm.android.utils

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
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

    @Test
    fun `buildLlmConfigLogPayload strips diffusion fields from llm payload`() {
        val source = """
            {
              "topK": 20,
              "system_prompt": "test",
              "diffusion_seed": 42,
              "diffusion_steps": 20,
              "image_width": 512,
              "image_height": 512,
              "cfg_prompt": "Generate high quality image",
              "grid_size": 1
            }
        """.trimIndent()

        val payload = CrashReportContext.buildLlmConfigLogPayload("load", source, maxConfigChars = 512)
        assertTrue(payload.contains("\"topK\":20"))
        assertTrue(payload.contains("\"system_prompt\":\"test\""))
        assertFalse(payload.contains("diffusion_seed"))
        assertFalse(payload.contains("diffusion_steps"))
        assertFalse(payload.contains("image_width"))
        assertFalse(payload.contains("image_height"))
        assertFalse(payload.contains("cfg_prompt"))
        assertFalse(payload.contains("grid_size"))
    }

    @Test
    fun `resolveCurrentActivityForCrashReport falls back to last known when current is blank`() {
        val resolved = CrashReportContext.resolveCurrentActivityForCrashReport(" ", "BenchmarkActivity")
        assertEquals("BenchmarkActivity", resolved)
    }

    @Test
    fun `buildBenchmarkContextLogPayload keeps stable defaults for missing values`() {
        val payload = CrashReportContext.buildBenchmarkContextLogPayload(
            modelId = null,
            backend = "",
            nPrompt = -1,
            nGenerate = 128,
            nRepeat = 0,
            kvCache = null
        )
        assertEquals(
            "[benchmark] model=none backend=cpu prompt=0 generate=128 repeat=1 kv_cache=false",
            payload
        )
    }

    @Test
    fun `extractLlmConfigCrashKeys reads use mmap and backend from standard llm keys`() {
        val source = """
            {
              "topK": 20,
              "backend_type": "opencl",
              "use_mmap": true
            }
        """.trimIndent()

        val keys = CrashReportContext.extractLlmConfigCrashKeys(source)
        assertEquals("opencl", keys.backend)
        assertTrue(keys.useMmap)
    }

    @Test
    fun `extractLlmConfigCrashKeys supports fallback aliases and defaults`() {
        val source = """
            {
              "backend": "cpu",
              "use_mma": "false"
            }
        """.trimIndent()

        val keys = CrashReportContext.extractLlmConfigCrashKeys(source)
        assertEquals("cpu", keys.backend)
        assertFalse(keys.useMmap)

        val defaultKeys = CrashReportContext.extractLlmConfigCrashKeys("{\"topK\":20}")
        assertEquals("unknown", defaultKeys.backend)
        assertFalse(defaultKeys.useMmap)
    }
}
