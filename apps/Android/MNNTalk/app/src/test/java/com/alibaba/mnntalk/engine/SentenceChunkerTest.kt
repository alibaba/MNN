package com.alibaba.mnntalk.engine

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class SentenceChunkerTest {
    @Test
    fun emitsStrongPunctuationImmediately() {
        val chunker = SentenceChunker()

        assertEquals(listOf("你好。"), chunker.append("你好。后半"))
        assertEquals("后半", chunker.flush())
    }

    @Test
    fun waitsForEnoughTextBeforeUsingComma() {
        val chunker = SentenceChunker(softLimit = 8)

        assertEquals(emptyList<String>(), chunker.append("你好，"))
        assertEquals(listOf("你好，这是较长，"), chunker.append("这是较长，"))
    }

    @Test
    fun stripsThinkingAndMarkdownBeforeSpeech() {
        val chunker = SentenceChunker()

        chunker.append("<think>hidden</think>**答案**")
        assertEquals("答案", chunker.flush())
        assertNull(chunker.flush())
    }
}
