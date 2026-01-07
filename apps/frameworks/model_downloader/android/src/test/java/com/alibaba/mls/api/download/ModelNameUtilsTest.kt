package com.alibaba.mls.api.download

import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

class ModelNameUtilsTest {

    @Test
    fun testGetModelName() {
        assertEquals("Qwen-1.5B", ModelNameUtils.getModelName("Huggingface/taobao-mnn/Qwen-1.5B"))
        assertEquals("simple-model", ModelNameUtils.getModelName("simple-model"))
        assertEquals("file.txt", ModelNameUtils.getModelName("path/to/file.txt"))
        assertNull(ModelNameUtils.getModelName(null))
        assertEquals("", ModelNameUtils.getModelName(""))
    }

    @Test
    fun testSafeModelId() {
        assertEquals("Huggingface_taobao-mnn_Qwen-1.5B", ModelNameUtils.safeModelId("Huggingface/taobao-mnn/Qwen-1.5B"))
    }

    @Test
    fun testSplitSource() {
        val parts = ModelNameUtils.splitSource("Huggingface/taobao-mnn/Qwen-1.5B")
        assertEquals(2, parts.size)
        assertEquals("Huggingface", parts[0])
        assertEquals("taobao-mnn/Qwen-1.5B", parts[1])
    }

    @Test
    fun testGetSource() {
        assertEquals("Huggingface", ModelNameUtils.getSource("Huggingface/taobao-mnn/Qwen-1.5B"))
        assertNull(ModelNameUtils.getSource("simple-model"))
        assertEquals("A", ModelNameUtils.getSource("A/B/C"))
    }

    @Test
    fun testGetRepositoryPath() {
        assertEquals("taobao-mnn/Qwen-1.5B", ModelNameUtils.getRepositoryPath("Huggingface/taobao-mnn/Qwen-1.5B"))
    }
}
