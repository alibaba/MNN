package com.alibaba.mnnllm.android.chat.chatlist

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class MarkdownBlockParserTest {

    @Test
    fun `splits fenced code into a horizontally scrollable block`() {
        val blocks = MarkdownBlockParser.parse(
            "Before\n\n```kotlin\nval longValue = someVeryLongFunctionName()\n```\n\nAfter",
            isStreaming = false
        )

        assertEquals(3, blocks.size)
        assertTrue(blocks[0] is MarkdownBlockParser.Block.Markdown)
        assertEquals(
            MarkdownBlockParser.Block.Code("val longValue = someVeryLongFunctionName()", "kotlin"),
            blocks[1]
        )
        assertTrue(blocks[2] is MarkdownBlockParser.Block.Markdown)
    }

    @Test
    fun `keeps a markdown table isolated from surrounding prose`() {
        val blocks = MarkdownBlockParser.parse(
            "Before\n\n| Name | Value |\n| --- | --- |\n| A | 1 |\n\nAfter",
            isStreaming = false
        )

        assertEquals(3, blocks.size)
        assertTrue(blocks[1] is MarkdownBlockParser.Block.Table)
        assertEquals(
            "| Name | Value |\n| --- | --- |\n| A | 1 |",
            blocks[1].content
        )
    }

    @Test
    fun `does not render an incomplete streaming table row`() {
        val blocks = MarkdownBlockParser.parse(
            "| Name | Value |\n| --- | --- |\n| incomplete",
            isStreaming = true
        )

        assertEquals(1, blocks.size)
        assertEquals(
            "| Name | Value |\n| --- | --- |",
            blocks[0].content
        )
    }

    @Test
    fun `supports tilde fences and an unfinished code block`() {
        val blocks = MarkdownBlockParser.parse(
            "~~~python\nprint('streaming')",
            isStreaming = true
        )

        assertEquals(
            listOf(MarkdownBlockParser.Block.Code("print('streaming')", "python")),
            blocks
        )
    }

    @Test
    fun `preserves one trailing newline without adding another blank line`() {
        val blocks = MarkdownBlockParser.parse("First line\n", isStreaming = false)

        assertEquals(
            listOf(MarkdownBlockParser.Block.Markdown("First line\n")),
            blocks
        )
    }
}
