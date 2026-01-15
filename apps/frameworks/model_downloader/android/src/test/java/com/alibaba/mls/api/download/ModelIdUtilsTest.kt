package com.alibaba.mls.api.download

import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class ModelIdUtilsTest {

    @Test
    fun testSafeModelId() {
        assertEquals("Huggingface_taobao-mnn_Qwen-1.5B", ModelIdUtils.safeModelId("Huggingface/taobao-mnn/Qwen-1.5B"))
        assertEquals("simple-model", ModelIdUtils.safeModelId("simple-model"))
        assertEquals("a_b_c", ModelIdUtils.safeModelId("a/b/c"))
        assertEquals("", ModelIdUtils.safeModelId(""))
    }

    @Test
    fun testSplitSource() {
        val parts1 = ModelIdUtils.splitSource("Huggingface/taobao-mnn/Qwen-1.5B")
        assertEquals(2, parts1.size)
        assertEquals("Huggingface", parts1[0])
        assertEquals("taobao-mnn/Qwen-1.5B", parts1[1])

        val parts2 = ModelIdUtils.splitSource("simple-model")
        assertEquals(1, parts2.size)
        assertEquals("simple-model", parts2[0])

        val parts3 = ModelIdUtils.splitSource("A/B/C")
        assertEquals(2, parts3.size)
        assertEquals("A", parts3[0])
        assertEquals("B/C", parts3[1])
    }

    @Test
    fun testGetRepositoryPath() {
        assertEquals("taobao-mnn/Qwen-1.5B", ModelIdUtils.getRepositoryPath("Huggingface/taobao-mnn/Qwen-1.5B"))
        assertEquals("simple-model", ModelIdUtils.getRepositoryPath("simple-model"))
        assertEquals("B/C", ModelIdUtils.getRepositoryPath("A/B/C"))
    }
}
