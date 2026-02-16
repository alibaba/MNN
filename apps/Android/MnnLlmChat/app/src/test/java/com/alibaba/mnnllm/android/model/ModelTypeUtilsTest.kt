package com.alibaba.mnnllm.android.model

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class ModelTypeUtilsTest {

    @Test
    fun `requiresFaceImageInput returns true for sana model`() {
        assertTrue(ModelTypeUtils.requiresFaceImageInput("local/sana-1600m"))
    }

    @Test
    fun `requiresFaceImageInput returns false for non sana model`() {
        assertFalse(ModelTypeUtils.requiresFaceImageInput("qwen2.5-7b-instruct"))
    }
}
