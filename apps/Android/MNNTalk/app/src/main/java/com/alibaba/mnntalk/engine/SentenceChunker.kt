package com.alibaba.mnntalk.engine

/**
 * Turns streamed LLM output into short, natural TTS segments.
 *
 * Strong punctuation is emitted immediately. Commas become boundaries only after
 * enough text has accumulated, which avoids both long TTS latency and choppy audio.
 */
class SentenceChunker(
    private val softLimit: Int = 36,
    private val hardLimit: Int = 72
) {
    private val buffer = StringBuilder()

    fun append(text: String): List<String> {
        buffer.append(text)
        val result = mutableListOf<String>()
        while (true) {
            val boundary = findBoundary() ?: break
            val segment = buffer.substring(0, boundary).trim()
            buffer.delete(0, boundary)
            if (segment.isNotEmpty()) {
                result += sanitize(segment)
            }
        }
        return result.filter { it.isNotBlank() }
    }

    fun flush(): String? {
        val segment = sanitize(buffer.toString().trim())
        buffer.clear()
        return segment.takeIf { it.isNotBlank() }
    }

    fun clear() {
        buffer.clear()
    }

    private fun findBoundary(): Int? {
        buffer.forEachIndexed { index, char ->
            if (char in STRONG_BOUNDARIES) {
                return index + 1
            }
            if (char in SOFT_BOUNDARIES && index + 1 >= softLimit) {
                return index + 1
            }
            if (index + 1 >= hardLimit) {
                return index + 1
            }
        }
        return null
    }

    private fun sanitize(text: String): String {
        return text
            .replace(THINK_BLOCK, "")
            .replace(MARKDOWN_MARKS, "")
            .replace(WHITESPACE, " ")
            .trim()
    }

    private companion object {
        val STRONG_BOUNDARIES = setOf('。', '！', '？', '!', '?', ';', '；', '\n')
        val SOFT_BOUNDARIES = setOf('，', ',', '、', '：', ':')
        val THINK_BLOCK = Regex("<think>[\\s\\S]*?</think>", RegexOption.IGNORE_CASE)
        val MARKDOWN_MARKS = Regex("[`*_#>]")
        val WHITESPACE = Regex("\\s+")
    }
}
