package com.alibaba.mnnllm.android.modelsettings

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class NumericInputParserTest {

    @Test
    fun `parseIntInput returns parsed value for valid integer`() {
        assertEquals(1024, parseIntInput("1024"))
    }

    @Test
    fun `parseIntInput returns null for empty input`() {
        assertNull(parseIntInput(""))
    }

    @Test
    fun `parseIntInput returns null for non numeric input`() {
        assertNull(parseIntInput("abc"))
    }

    @Test
    fun `parseIntInput returns null for int overflow input`() {
        assertNull(parseIntInput("9999999999"))
    }
}
