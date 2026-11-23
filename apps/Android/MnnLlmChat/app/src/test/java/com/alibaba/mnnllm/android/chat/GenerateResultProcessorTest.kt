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
    fun answerDisplayText_allChunkSplitPoints_hideOnlyOuterMarkers() {
        val body = "完整回答 42\n第二行。"
        val opening = "<answer>"
        val input = opening + body + "</answer>"
        for (first in 0..input.length) {
            for (second in first..input.length) {
                val processor = GenerateResultProcessor()
                processor.generateBegin()
                var previous = 0
                for (end in listOf(first, second, input.length)) {
                    processor.process(input.substring(previous, end))
                    val expected = body.take((end - opening.length).coerceIn(0, body.length))
                    assertEquals("splits=$first,$second end=$end", expected,
                        GenerateResultProcessor.answerDisplayText(processor.getNormalOutput(), streaming = true))
                    assertEquals(input.substring(0, end), processor.getRawResult())
                    assertEquals("", processor.getThinkingContent())
                    previous = end
                }
                processor.process(null)
                assertEquals(body, GenerateResultProcessor.answerDisplayText(processor.getNormalOutput()))
                assertEquals(input, processor.getRawResult())
            }
        }
    }

    @Test
    fun answerDisplayText_preservesWhitespaceAndInnerLiteralTags() {
        val body = "\n正文 <answer>字面示例</answer> 仍然保留\n"
        assertEquals(" \t" + body + "\r\n", GenerateResultProcessor.answerDisplayText(
            " \t<answer>" + body + "</answer>\r\n"))
        assertEquals("未完成回答", GenerateResultProcessor.answerDisplayText("<answer>未完成回答"))
        assertEquals("<answer>示例</answer>", GenerateResultProcessor.answerDisplayText(
            "<answer><answer>示例</answer></answer>"))
        assertEquals("正文 </answer> 中间字面结束标签", GenerateResultProcessor.answerDisplayText(
            "<answer>正文 </answer> 中间字面结束标签</answer>"))
        assertEquals("正文 <answer> 表示开始", GenerateResultProcessor.answerDisplayText(
            "<answer>正文 <answer> 表示开始</answer>"))
        assertEquals(" \n", GenerateResultProcessor.answerDisplayText(" \n<ans", streaming = true))
        assertEquals(" \n<ans", GenerateResultProcessor.answerDisplayText(" \n<ans"))
    }

    @Test
    fun answerDisplayText_bareTextAllSplitPoints_isNotLost() {
        val inputs = listOf(
            "裸文本回答\n 保留空格 ", "<", "<a", "<ans", "<answer", "</answer>",
            "<answerX>不是控制标记</answerX>", "<answer attr=\"x\">保留</answer>",
            "<ANSWER>保留</ANSWER>", "正文 <answer>示例</answer>",
            "`<answer>示例</answer>`", "```xml\n<answer>示例</answer>\n```",
            "~~~xml\n<answer>示例</answer>\n~~~", "<xml>保留</xml>"
        )
        for (input in inputs) {
            for (split in 0..input.length) {
                val processor = GenerateResultProcessor()
                processor.generateBegin()
                val prefix = input.substring(0, split)
                processor.process(prefix)
                val openingCandidates = listOf("<", "<a", "<an", "<ans", "<answ", "<answe", "<answer")
                assertEquals(if (prefix in openingCandidates) "" else prefix,
                    GenerateResultProcessor.answerDisplayText(processor.getNormalOutput(), streaming = true))
                processor.process(input.substring(split))
                processor.process(null)
                assertEquals("input=$input split=$split", input,
                    GenerateResultProcessor.answerDisplayText(processor.getNormalOutput()))
                assertEquals(input, processor.getRawResult())
            }
        }
        assertEquals("", GenerateResultProcessor.answerDisplayText(null))
        assertEquals("", GenerateResultProcessor.answerDisplayText("<ans", streaming = true))
        assertEquals("正文", GenerateResultProcessor.answerDisplayText("<answer>正文</ans", streaming = true))
        assertEquals("正文</ans", GenerateResultProcessor.answerDisplayText("<answer>正文</ans"))
        assertEquals("</ans", GenerateResultProcessor.answerDisplayText("</ans", streaming = true))
    }

    @Test
    fun answerDisplayText_wrappedCode_keepsLiteralTagsAndPartialSuffixes() {
        val bodies = listOf(
            "`<answer>示例</answer>`", "``literal `<answer>示例</answer>` ``",
            "```xml\n<answer>示例</answer>\n```", "~~~xml\n<answer>示例</answer>\n~~~",
            "```xml\n<answer>示例</answer>\n```\n", "~~~xml\n<answer>示例</answer>\n~~~\n",
            "  ````xml\n<answer>示例</answer>\n```\n````\n",
            "```xml\r<answer>示例</answer>\r```\r", "\\`不是 inline code"
        )
        for (body in bodies) {
            val input = "<answer>$body</answer>"
            for (end in 0..input.length) {
                val expected = body.take((end - "<answer>".length).coerceIn(0, body.length))
                assertEquals("body=$body end=$end", expected,
                    GenerateResultProcessor.answerDisplayText(input.substring(0, end), streaming = true))
            }
            assertEquals(body, GenerateResultProcessor.answerDisplayText(input))
        }
        for (body in listOf("`字面 </answer>", "```xml\n字面 </answer>", "~~~xml\n字面 </answer>")) {
            assertEquals(body, GenerateResultProcessor.answerDisplayText("<answer>$body"))
            for (length in 1 until "</answer>".length) {
                val partial = body.removeSuffix("</answer>") + "</answer>".take(length)
                assertEquals(partial, GenerateResultProcessor.answerDisplayText("<answer>$partial", streaming = true))
            }
        }
    }

    @Test
    fun answerDisplayText_qwenThinkParsing_isUnchanged() {
        val answer = "正常回答\n`<answer>字面示例</answer>`"
        for (split in 0..answer.length) {
            val processor = GenerateResultProcessor()
            processor.generateBegin()
            processor.process("<think>")
            processor.process("先思考")
            processor.process("再确认")
            processor.process("</think>")
            processor.process(answer.substring(0, split))
            processor.process(answer.substring(split))
            processor.process(null)
            assertEquals("先思考再确认\n", processor.getThinkingContent())
            assertEquals(answer, processor.getNormalOutput())
            assertEquals(answer, GenerateResultProcessor.answerDisplayText(processor.getNormalOutput()))
            assertEquals("先思考再确认\n$answer", processor.getDisplayResult())
            assertEquals("<think>先思考再确认</think>$answer", processor.getRawResult())
            assertTrue(processor.thinkTime >= 0)
        }
        val raw = "<think>保留现有解析职责</think>正常回答"
        assertEquals(raw, GenerateResultProcessor.answerDisplayText(raw))
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

