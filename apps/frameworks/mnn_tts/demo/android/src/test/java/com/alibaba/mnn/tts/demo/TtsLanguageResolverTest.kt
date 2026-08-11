package com.alibaba.mnn.tts.demo

import org.junit.Assert.assertEquals
import org.junit.Test

class TtsLanguageResolverTest {
    @Test
    fun `returns zh when text contains chinese characters`() {
        assertEquals("zh", TtsLanguageResolver.resolve("你好，今天天气不错。", "en"))
    }

    @Test
    fun `returns selected language for non chinese text`() {
        assertEquals("en", TtsLanguageResolver.resolve("Hello, this is a test.", "en"))
    }

    @Test
    fun `keeps zh fallback when selected language is empty`() {
        assertEquals("zh", TtsLanguageResolver.resolve("", ""))
    }
}
