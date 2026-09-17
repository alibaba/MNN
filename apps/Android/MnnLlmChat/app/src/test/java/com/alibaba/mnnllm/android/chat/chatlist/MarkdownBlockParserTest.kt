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

    @Test
    fun `pads a delimiter row that has fewer cells than the header`() {
        // GFM rejects such a table outright, which would degrade it into plain text.
        val blocks = MarkdownBlockParser.parse(
            "| A | B | C |\n|---|---|\n| 1 | 2 | 3 |",
            isStreaming = false
        )

        assertEquals(1, blocks.size)
        assertTrue(blocks[0] is MarkdownBlockParser.Block.Table)
        assertEquals(
            "| A | B | C |\n| --- | --- | --- |\n| 1 | 2 | 3 |",
            blocks[0].content
        )
    }

    @Test
    fun `trims a delimiter row that has more cells than the header`() {
        val blocks = MarkdownBlockParser.parse(
            "| A | B |\n|---|---|---|---|\n| 1 | 2 |",
            isStreaming = false
        )

        assertEquals(
            "| A | B |\n| --- | --- |\n| 1 | 2 |",
            blocks[0].content
        )
    }

    @Test
    fun `parses table rows and drops the delimiter row`() {
        val rows = MarkdownBlockParser.parseTableRows(
            "| A | B |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |"
        )

        assertEquals(
            listOf(listOf("A", "B"), listOf("1", "2"), listOf("3", "4")),
            rows
        )
    }

    @Test
    fun `normalises row length against the header`() {
        val rows = MarkdownBlockParser.parseTableRows(
            "| A | B | C |\n| --- | --- | --- |\n| 1 |\n| 1 | 2 | 3 | 4 |"
        )

        assertEquals(listOf("1", "", ""), rows[1])
        assertEquals(listOf("1", "2", "3"), rows[2])
    }

    @Test
    fun `keeps an escaped pipe as cell content`() {
        val rows = MarkdownBlockParser.parseTableRows(
            "| Content | Note |\n| --- | --- |\n| a\\|b | normal |"
        )

        assertEquals(listOf("Content", "Note"), rows[0])
        assertEquals(listOf("a|b", "normal"), rows[1])
    }

    @Test
    fun `accepts a delimiter row with a single dash per column`() {
        val blocks = MarkdownBlockParser.parse("| A | B |\n|-|-|\n| 1 | 2 |", isStreaming = false)

        assertTrue(blocks[0] is MarkdownBlockParser.Block.Table)
    }

    @Test
    fun `accepts compact alignment markers`() {
        val blocks = MarkdownBlockParser.parse("| A | B |\n|:-:|:-:|\n| 1 | 2 |", isStreaming = false)

        assertTrue(blocks[0] is MarkdownBlockParser.Block.Table)
    }

    @Test
    fun `accepts a single column table`() {
        val blocks = MarkdownBlockParser.parse("| A |\n| --- |\n| 1 |", isStreaming = false)

        assertTrue(blocks[0] is MarkdownBlockParser.Block.Table)
    }

    @Test
    fun `does not treat a bare thematic break as a delimiter row`() {
        val blocks = MarkdownBlockParser.parse("A | B\n---\ntext", isStreaming = false)

        assertTrue(blocks.none { it is MarkdownBlockParser.Block.Table })
    }


    @Test
    fun `never flips a table back to prose while streaming`() {
        // Every flip makes MarkdownMessageView rebuild the whole message, which is the flicker
        // this parser is meant to avoid. Walk every prefix of an aligned table and assert the
        // block type only ever advances from prose to table.
        val full = "| A | B |\n| :-: | :-: |\n| 1 | 2 |"
        val transitions = mutableListOf<String>()
        for (length in 1..full.length) {
            val blocks = MarkdownBlockParser.parse(full.substring(0, length), isStreaming = true)
            val kind = if (blocks.any { it is MarkdownBlockParser.Block.Table }) "table" else "prose"
            if (transitions.isEmpty() || transitions.last() != kind) {
                transitions.add(kind)
            }
        }

        assertEquals(listOf("prose", "table"), transitions)
    }

    @Test
    fun `returns no rows for content without cells`() {
        assertEquals(emptyList<List<String>>(), MarkdownBlockParser.parseTableRows(""))
    }

    @Test
    fun `keeps alignment markers when padding a delimiter row`() {
        val blocks = MarkdownBlockParser.parse(
            "| A | B | C |\n| :--- | ---: |\n| 1 | 2 | 3 |",
            isStreaming = false
        )

        assertEquals(
            "| A | B | C |\n| :--- | ---: | --- |\n| 1 | 2 | 3 |",
            blocks[0].content
        )
    }
}
