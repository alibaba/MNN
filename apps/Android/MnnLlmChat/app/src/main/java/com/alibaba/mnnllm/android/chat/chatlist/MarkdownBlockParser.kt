package com.alibaba.mnnllm.android.chat.chatlist

internal object MarkdownBlockParser {
    sealed class Block(open val content: String) {
        data class Markdown(override val content: String) : Block(content)
        data class Table(override val content: String) : Block(content)
        data class Code(override val content: String, val language: String?) : Block(content)
    }

    /** Index of the `|---|---|` row inside a table block. */
    private const val DELIMITER_LINE_INDEX = 1
    private const val DEFAULT_ALIGNMENT = "---"

    /** A cell separator, i.e. a `|` that is not escaped as `\|`. */
    private val cellSeparator = Regex("""(?<!\\)\|""")

    private val openingFence = Regex("""^ {0,3}(`{3,}|~{3,})(.*)$""")
    /**
     * A GFM delimiter row: each cell is one or more `-` with optional leading/trailing `:` for
     * alignment. A single column is valid, and one dash per cell is enough (`|-|-|`).
     */
    private val tableSeparator = Regex(
        """^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$"""
    )

    /** A delimiter row that is still being streamed: only separator characters so far. */
    private val partialTableSeparator = Regex("""^[\s|:-]+$""")

    fun parse(text: String, isStreaming: Boolean): List<Block> {
        if (text.isEmpty()) return emptyList()

        val endsWithNewLine = text.endsWith('\n')
        val lines = if (endsWithNewLine) {
            text.dropLast(1).split('\n')
        } else {
            text.split('\n')
        }
        val blocks = mutableListOf<Block>()
        val markdown = StringBuilder()
        var lineIndex = 0

        fun appendLine(builder: StringBuilder, line: String, index: Int) {
            builder.append(line)
            if (index < lines.lastIndex || endsWithNewLine) {
                builder.append('\n')
            }
        }

        fun flushMarkdown() {
            if (markdown.isNotEmpty()) {
                blocks += Block.Markdown(markdown.toString())
                markdown.clear()
            }
        }

        while (lineIndex < lines.size) {
            val line = lines[lineIndex]
            val fenceMatch = openingFence.matchEntire(line)
            if (fenceMatch != null) {
                flushMarkdown()
                val fence = fenceMatch.groupValues[1]
                val language = fenceMatch.groupValues[2]
                    .trim()
                    .substringBefore(' ')
                    .ifEmpty { null }
                val code = StringBuilder()
                lineIndex += 1
                while (lineIndex < lines.size) {
                    val codeLine = lines[lineIndex]
                    if (isClosingFence(codeLine, fence)) {
                        lineIndex += 1
                        break
                    }
                    appendLine(code, codeLine, lineIndex)
                    lineIndex += 1
                }
                blocks += Block.Code(code.toString().removeSuffix("\n"), language)
                continue
            }

            if (isTableStart(lines, lineIndex, isStreaming)) {
                flushMarkdown()
                val table = StringBuilder()
                val headerLine = lines[lineIndex]
                appendLine(table, headerLine, lineIndex)
                appendLine(table, alignSeparator(headerLine, lines[lineIndex + 1]), lineIndex + 1)
                lineIndex += 2

                while (lineIndex < lines.size && looksLikeTableRow(lines[lineIndex])) {
                    val trailingIncompleteRow = isStreaming && !endsWithNewLine && lineIndex == lines.lastIndex
                    if (trailingIncompleteRow) {
                        lineIndex += 1
                        break
                    }
                    appendLine(table, lines[lineIndex], lineIndex)
                    lineIndex += 1
                }
                blocks += Block.Table(table.toString().removeSuffix("\n"))
                continue
            }

            appendLine(markdown, line, lineIndex)
            lineIndex += 1
        }

        flushMarkdown()
        return blocks
    }

    fun splitCells(line: String): List<String> {
        val cells = line.trim().split(cellSeparator).toMutableList()
        // A leading or trailing separator yields an empty element on that edge.
        if (cells.size > 1 && cells.first().isEmpty()) {
            cells.removeAt(0)
        }
        if (cells.size > 1 && cells.last().isEmpty()) {
            cells.removeAt(cells.lastIndex)
        }
        return cells.map { it.trim().replace("\\|", "|") }
    }

    /**
     * Splits a table block into cell rows, dropping the delimiter row and normalising every row
     * to the header's column count the way GFM does (pad short rows, truncate long ones).
     */
    fun parseTableRows(content: String): List<List<String>> {
        val rows = content.lineSequence()
            .filterIndexed { index, line -> index != DELIMITER_LINE_INDEX && line.contains('|') }
            .map { splitCells(it) }
            .toList()
        if (rows.isEmpty()) return emptyList()
        val columnCount = rows[0].size
        return rows.map { row ->
            when {
                row.size == columnCount -> row
                row.size < columnCount -> row + List(columnCount - row.size) { "" }
                else -> row.take(columnCount)
            }
        }
    }

    /**
     * GFM only treats a block as a table when the delimiter row has exactly as many cells as the
     * header row, and models routinely emit one cell too few. Without this the whole table would
     * silently degrade into a wall of plain text.
     */
    private fun alignSeparator(headerLine: String, separatorLine: String): String {
        val headerCellCount = splitCells(headerLine).size
        val separatorCells = splitCells(separatorLine)
        if (separatorCells.size == headerCellCount) {
            return separatorLine
        }
        return (0 until headerCellCount)
            .map { index -> separatorCells.getOrNull(index)?.takeIf { it.isNotEmpty() } ?: DEFAULT_ALIGNMENT }
            .joinToString(separator = " | ", prefix = "| ", postfix = " |")
    }

    private fun isTableStart(lines: List<String>, index: Int, isStreaming: Boolean): Boolean {
        if (index + 1 >= lines.size || !lines[index].contains('|')) return false
        val delimiter = lines[index + 1]
        // Requiring a pipe keeps a bare `---` (thematic break, setext heading) from being read
        // as a single-column delimiter row.
        if (!delimiter.contains('|')) return false
        if (tableSeparator.matches(delimiter)) return true
        // While streaming, the delimiter row arrives character by character, so states such as
        // `| :-: | :` are not valid yet. Treating them as prose would flip the block type back to
        // Markdown and rebuild the whole message on the next token, so accept a partial delimiter
        // as long as it is still the last line.
        return isStreaming &&
                index + 1 == lines.lastIndex &&
                partialTableSeparator.matches(delimiter)
    }

    private fun looksLikeTableRow(line: String): Boolean {
        return line.isNotBlank() && line.contains('|')
    }

    private fun isClosingFence(line: String, openingFence: String): Boolean {
        val fenceChar = Regex.escape(openingFence.first().toString())
        return Regex("""^ {0,3}$fenceChar{${openingFence.length},}\s*$""").matches(line)
    }
}
