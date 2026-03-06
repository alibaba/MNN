package com.alibaba.mnnllm.android.chat

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class DiffusionProgressHintPolicyTest {

    @Test
    fun `marks positive integer progress as meaningful`() {
        assertTrue(DiffusionProgressHintPolicy.isMeaningfulProgress("1"))
        assertTrue(DiffusionProgressHintPolicy.isMeaningfulProgress("37"))
    }

    @Test
    fun `treats zero null and invalid progress as not meaningful`() {
        assertFalse(DiffusionProgressHintPolicy.isMeaningfulProgress("0"))
        assertFalse(DiffusionProgressHintPolicy.isMeaningfulProgress(null))
        assertFalse(DiffusionProgressHintPolicy.isMeaningfulProgress("abc"))
    }
}
