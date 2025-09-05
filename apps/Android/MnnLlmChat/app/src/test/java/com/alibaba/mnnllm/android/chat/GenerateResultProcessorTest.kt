package com.alibaba.mnnllm.android.chat

import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class GenerateResultProcessorTest {

    @Test
    fun noSlashThink_removesLeadingEndTag() {
        assertEquals("abc", GenerateResultProcessor.noSlashThink("</think>abc"))
        assertEquals("xyz</think>", GenerateResultProcessor.noSlashThink("xyz</think>"))
        assertNull(GenerateResultProcessor.noSlashThink(null))
    }

    @Test
    fun thinkTags_simpleBlock_parsesThinkingAndNormal() {
        val p = GenerateResultProcessor()
        p.generateBegin()

        p.process("<think>abc</think>def")
        p.process(null)

        // THINK_TAGS adds a trailing newline on think end
        assertEquals("abc\n", p.getThinkingContent())
        assertEquals("def", p.getNormalOutput())
        assertEquals("abc\ndef", p.getDisplayResult())
        assertTrue(p.thinkTime >= 0)
    }

    @Test
    fun thinkTags_emptyThink_hasNoThinkingContent() {
        val p = GenerateResultProcessor()
        p.generateBegin()

        p.process("<think></think>abc")
        p.process(null)

        // No content inside <think> -> thinking content should be empty string
        assertEquals("", p.getThinkingContent())
        assertEquals("abc", p.getNormalOutput())
        assertEquals("abc", p.getDisplayResult())
        assertTrue(p.thinkTime >= 0)
    }

    @Test
    fun thinkTags_retroactiveMove_onEndTagFirst() {
        val p = GenerateResultProcessor()
        p.generateBegin()

        // Stream normal text first, then an unexpected </think> which should
        // retroactively move the pending normal text into thinking.
        p.process("Hello ")
        p.process("World")
        p.process("</think> Rest")
        p.process(null)

        assertEquals("Hello World\n", p.getThinkingContent())
        assertEquals(" Rest", p.getNormalOutput())
        assertEquals("Hello World\n Rest", p.getDisplayResult())
        assertTrue(p.thinkTime >= 0)
    }

    @Test
    fun gptOss_thinkingOnly_extractsMessage() {
        val p = GenerateResultProcessor()
        p.generateBegin()

        // Contains a <|message|> block but no final<|message|>
        p.process("prefix<|message|>thinking...<|end|>suffix")
        // End of stream should set thinkTime if not already set
        p.process(null)

        assertEquals("thinking...", p.getThinkingContent())
        assertEquals("", p.getNormalOutput())
        assertEquals("thinking...", p.getDisplayResult())
        assertTrue(p.thinkTime >= 0)
    }

    @Test
    fun gptOss_withFinalMessage_splitsThinkingAndNormal() {
        val p = GenerateResultProcessor()
        p.generateBegin()

        // The last occurrence of "final<|message|>" marks the start of normal output
        val input = "noise<|message|>internal-think<|end|>final<|message|>Hello world"
        p.process(input)
        p.process(null)

        assertEquals("internal-think", p.getThinkingContent())
        assertEquals("Hello world", p.getNormalOutput())
        assertEquals("internal-thinkHello world", p.getDisplayResult())
        assertTrue(p.thinkTime >= 0)
    }
}

