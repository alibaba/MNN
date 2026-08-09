package com.alibaba.mnnllm.android.chat.chatlist

internal object MarkdownBlockParser {
    sealed class Block(open val content: String) {
        data class Markdown(override val content: String) : Block(content)
        data class Table(override val content: String) : Block(content)
        data class Code(override val content: String, val language: String?) : Block(content)
    }

    private val openingFence = Regex("""^ {0,3}(`{3,}|~{3,})(.*)$""")
    private val tableSeparator = Regex(
        """^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$"""
    )

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

            if (isTableStart(lines, lineIndex)) {
                flushMarkdown()
                val table = StringBuilder()
                appendLine(table, lines[lineIndex], lineIndex)
                appendLine(table, lines[lineIndex + 1], lineIndex + 1)
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

    private fun isTableStart(lines: List<String>, index: Int): Boolean {
        if (index + 1 >= lines.size || !lines[index].contains('|')) return false
        return tableSeparator.matches(lines[index + 1])
    }

    private fun looksLikeTableRow(line: String): Boolean {
        return line.isNotBlank() && line.contains('|')
    }

    private fun isClosingFence(line: String, openingFence: String): Boolean {
        val fenceChar = Regex.escape(openingFence.first().toString())
        return Regex("""^ {0,3}$fenceChar{${openingFence.length},}\s*$""").matches(line)
    }
}
