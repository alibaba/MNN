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

    @Test
    fun `isOpenClWarningByExtraTags returns true when opencl warning tag exists`() {
        assertTrue(ModelTypeUtils.isOpenClWarningByExtraTags(listOf("opencl_warning")))
        assertTrue(ModelTypeUtils.isOpenClWarningByExtraTags(listOf("OpenCLWarning")))
    }

    @Test
    fun `isOpenClWarningByExtraTags returns false when tag does not exist`() {
        assertFalse(ModelTypeUtils.isOpenClWarningByExtraTags(listOf("ThinkingSwitch")))
    }

}
