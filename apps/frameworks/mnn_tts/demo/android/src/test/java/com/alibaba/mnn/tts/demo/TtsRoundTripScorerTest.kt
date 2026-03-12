package com.alibaba.mnn.tts.demo

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class TtsRoundTripScorerTest {
    @Test
    fun `passes when recognized chinese text matches after normalization`() {
        val result = TtsRoundTripScorer.score(
            expected = "你好，今天天气不错。",
            actual = "你好 今天天气不错"
        )

        assertTrue(result.hasChinese)
        assertTrue(result.similarity > 0.9)
        assertEquals("PASS", result.status)
    }

    @Test
    fun `fails when recognized text has no chinese characters`() {
        val result = TtsRoundTripScorer.score(
            expected = "你好，今天天气不错。",
            actual = "ni hao"
        )

        assertFalse(result.hasChinese)
        assertEquals("FAIL", result.status)
    }

    @Test
    fun `fails when chinese roundtrip similarity is too low`() {
        val result = TtsRoundTripScorer.score(
            expected = "你好，今天天气不错。",
            actual = "可以滚"
        )

        assertTrue(result.hasChinese)
        assertEquals(0.0, result.similarity, 0.0)
        assertEquals("FAIL", result.status)
    }
}
