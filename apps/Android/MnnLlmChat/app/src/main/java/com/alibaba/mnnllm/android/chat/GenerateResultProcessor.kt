// Created by ruoyi.sjd on 2025/2/6.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.chat

import android.util.Log
import java.lang.StringBuilder

/**
 * Processes LLM generation output streams, supporting two formats:
 * 1. THINK_TAGS: Uses <think> and </think> to denote thinking sections.
 * 2. GPT_OSS: Uses specific control tags like "<|channel|>" and "<|message|>"
 * to structure the output.
 *
 * The class auto-detects the format based on the initial stream content.
 */
class GenerateResultProcessor {

    companion object {
        const val TAG: String = "GenerateResultProcessor"

        /**
         * Helper function to remove a leftover </think> tag from the beginning of a string,
         * which can happen if the stream is cut off right after the tag.
         */
        fun noSlashThink(text: String?): String? {
            if (text?.startsWith("</think>") == true) {
                return text.substring("</think>".length)
            }
            return text
        }
    }

    // Enum to represent the detected stream format
    private enum class StreamFormat {
        UNKNOWN,
        THINK_TAGS,
        GPT_OSS
    }

    // General state
    private var generateBeginTime: Long = 0
    private var currentFormat: StreamFormat = StreamFormat.UNKNOWN

    // Stores the complete raw input for debugging and for GPT_OSS processing
    private val rawStringBuilder = StringBuilder()

    // Final output builders, used by both formats
    val thinkingStringBuilder = StringBuilder()
    val normalStringBuilder = StringBuilder()
    var thinkTime: Long = -1L
        private set

    // --- State for THINK_TAGS format ---
    private var isThinking: Boolean = false
    private var hasThought: Boolean = false
    private var thinkHasContent: Boolean = false
    private var tagBuffer = "" // Buffer for handling tags split across chunks
    private val pendingTextBuffer = StringBuilder() // Buffer for text that might be normal or thinking

    /**
     * Resets the processor's state for a new generation session.
     */
    fun reset() {
        generateBeginTime = 0
        currentFormat = StreamFormat.UNKNOWN
        rawStringBuilder.clear()
        thinkingStringBuilder.clear()
        normalStringBuilder.clear()
        pendingTextBuffer.clear()
        thinkTime = -1L
        isThinking = false
        hasThought = false
        thinkHasContent = false
        tagBuffer = ""
    }

    /**
     * Call this when the generation process begins to set the start time.
     */
    fun generateBegin() {
        reset() // Reset state before starting
        this.generateBeginTime = System.currentTimeMillis()
        Log.d(TAG, "generateBegin ${this.generateBeginTime}")
    }

    /**
     * Processes a chunk of the LLM output stream.
     *
     * @param progress The incoming string chunk, or null if the stream has ended.
     */
    fun process(progress: String?) {
        Log.d(TAG, "process: #${progress}#")
        if (progress == null) {
            // Handle end of stream based on the detected format
            when (currentFormat) {
                StreamFormat.THINK_TAGS, StreamFormat.UNKNOWN -> processThinkTags(null)
                StreamFormat.GPT_OSS -> processGptOss(null)
            }
            return
        }

        // Append to the raw string builder first. This is crucial for GPT_OSS logic
        // which re-processes the entire history.
        rawStringBuilder.append(progress)

        // --- Format Detection ---
        if (currentFormat == StreamFormat.UNKNOWN) {
            // GPT_OSS tags are very specific and a strong indicator.
            if (rawStringBuilder.contains("<|message|>") || rawStringBuilder.contains("<|channel|>")) {
                currentFormat = StreamFormat.GPT_OSS
                Log.d(TAG, "Format detected: GPT_OSS. Reprocessing entire buffer.")
                // Since we've detected the format, we must re-process the entire buffered content
                // with the correct logic.
                processGptOss(rawStringBuilder.toString())
                return // Return because the entire buffer has been processed by the new logic
            } else if (rawStringBuilder.contains("<think>") || rawStringBuilder.contains("</think>")) {
                currentFormat = StreamFormat.THINK_TAGS
                Log.d(TAG, "Format detected: THINK_TAGS")
            }
        }

        // --- Delegate to the appropriate processor ---
        when (currentFormat) {
            StreamFormat.GPT_OSS -> {
                // For subsequent chunks, we just re-process the whole raw string.
                processGptOss(rawStringBuilder.toString())
            }
            // Default to THINK_TAGS logic if format is THINK_TAGS or still UNKNOWN
            else -> {
                processThinkTags(progress)
            }
        }
    }

    // ===================================================================
    //  LOGIC FOR GPT_OSS FORMAT (<|message|> delimiter)
    // ===================================================================

    private fun processGptOss(fullInput: String?) {
        if (fullInput == null) {
            if (thinkTime == -1L && rawStringBuilder.isNotEmpty()) {
                thinkTime = System.currentTimeMillis() - generateBeginTime
            }
            return
        }
        val finalMessageDelimiter = "final<|message|>"
        thinkingStringBuilder.clear()
        normalStringBuilder.clear()
        val finalMessageIndex = fullInput.lastIndexOf(finalMessageDelimiter)
        if (finalMessageIndex != -1) {
            val normalContent = fullInput.substring(finalMessageIndex + finalMessageDelimiter.length)
            normalStringBuilder.append(normalContent)
            val rawThinkBlock = fullInput.substring(0, finalMessageIndex)
            parseGptOssThinkingBlock(rawThinkBlock)
            if (thinkTime == -1L) {
                thinkTime = System.currentTimeMillis() - generateBeginTime
                Log.d(TAG, "GPT_OSS: Final message boundary found. Think time: $thinkTime")
            }
        } else {
            parseGptOssThinkingBlock(fullInput)
        }
    }

    private fun parseGptOssThinkingBlock(block: String) {
        val thinkMessageStartTag = "<|message|>"
        val thinkMessageEndTag = "<|end|>"
        val firstMessageIndex = block.indexOf(thinkMessageStartTag)
        if (firstMessageIndex != -1) {
            val endTagIndex = block.indexOf(thinkMessageEndTag, startIndex = firstMessageIndex)
            val thinkContent = if (endTagIndex != -1) {
                block.substring(firstMessageIndex + thinkMessageStartTag.length, endTagIndex).trim()
            } else {
                block.substring(firstMessageIndex + thinkMessageStartTag.length).trim()
            }
            formatAndSetGptOssThinkingContent(thinkContent)
        }
    }

    private fun formatAndSetGptOssThinkingContent(content: String) {
        if (content.isNotBlank()) {
            thinkingStringBuilder.append("\n> ")
            thinkingStringBuilder.append(content.replace("\n", "\n> "))
            thinkingStringBuilder.append("\n")
        }
    }

    // ===================================================================
    //  LOGIC FOR THINK_TAGS FORMAT (<think>...</think>)
    // ===================================================================

    private fun processThinkTags(progress: String?) {
        if (progress == null) {
            // End of stream.
            // 1. Flush any incomplete tag text from the tag buffer as normal text.
            if (tagBuffer.isNotEmpty()) {
                normalStringBuilder.append(tagBuffer)
                tagBuffer = ""
            }
            // 2. Any text held in the pending buffer is now confirmed as normal.
            //    The text is already in normalStringBuilder, so we just clear the pending
            //    buffer to prevent any further (and incorrect) retroactive changes.
            pendingTextBuffer.clear()

            // 3. If we were in the middle of a think block, force it to close.
            if (isThinking) {
                handleThinkEnd(force = true)
            }
            return
        }

        var buffer = tagBuffer + progress
        tagBuffer = ""

        while (buffer.isNotEmpty()) {
            val thinkTag = "<think>"
            val endThinkTag = "</think>"
            val thinkIndex = buffer.indexOf(thinkTag)
            val endThinkIndex = buffer.indexOf(endThinkTag)

            if (isThinking) {
                // --- We are INSIDE a think block ---
                val effectiveEndIndex = if (endThinkIndex != -1) endThinkIndex else buffer.length
                val text = buffer.substring(0, effectiveEndIndex)

                if (text.isNotEmpty()) {
                    thinkingStringBuilder.append(text.replace("\n", "\n> "))
                    thinkHasContent = true
                }

                if (endThinkIndex != -1) {
                    handleThinkEnd()
                    buffer = buffer.substring(endThinkIndex + endThinkTag.length)
                } else {
                    buffer = "" // Consumed entire buffer as thinking text
                }
            } else {
                // --- We are OUTSIDE a think block ---
                if (thinkIndex != -1 && (endThinkIndex == -1 || thinkIndex < endThinkIndex)) {
                    // Case A: Start tag appears first.
                    val textBefore = buffer.substring(0, thinkIndex)

                    // The text before the <think> tag is definitely normal.
                    normalStringBuilder.append(textBefore)
                    // Since we found a tag, the pending block is confirmed and can be cleared.
                    pendingTextBuffer.clear()

                    handleThinkStart()
                    buffer = buffer.substring(thinkIndex + thinkTag.length)

                } else if (endThinkIndex != -1) {
                    // Case B: End tag appears first.
                    val textBefore = buffer.substring(0, endThinkIndex)

                    // The text in the pending buffer and the text before the end tag
                    // must be retroactively moved to the thinking section.

                    // 1. Remove the pending text from the normal output.
                    if (pendingTextBuffer.isNotEmpty()) {
                        val start = normalStringBuilder.length - pendingTextBuffer.length
                        if (start >= 0) {
                            normalStringBuilder.delete(start, normalStringBuilder.length)
                        }
                    }

                    // 2. Start the thinking block in the output.
                    handleThinkStart()

                    // 3. Add the pending text and the current text to the thinking block.
                    val textToThink = pendingTextBuffer.toString() + textBefore
                    if (textToThink.isNotEmpty()) {
                        thinkingStringBuilder.append(textToThink.replace("\n", "\n> "))
                        thinkHasContent = true
                    }

                    // 4. Clear the pending buffer and end the thinking block.
                    pendingTextBuffer.clear()
                    handleThinkEnd()

                    // 5. Move the buffer past the processed tag.
                    buffer = buffer.substring(endThinkIndex + endThinkTag.length)

                } else {
                    // Case C: No tags found in this buffer.
                    // Treat it as potentially normal text for now, and track it.
                    normalStringBuilder.append(buffer)
                    pendingTextBuffer.append(buffer)
                    buffer = ""
                }
            }
        }
    }

    private fun handleThinkStart() {
        if (!isThinking) {
            isThinking = true
            if (!hasThought) {
                hasThought = true
                thinkingStringBuilder.append("\n> ")
            } else {
                thinkingStringBuilder.append("\n> ") // Separator for subsequent thoughts
            }
        }
    }

    private fun handleThinkEnd(force: Boolean = false) {
        if (isThinking || force) {
            isThinking = false
            if (thinkTime == -1L) {
                thinkTime = System.currentTimeMillis() - generateBeginTime
                Log.d(TAG, "THINK_TAGS: </think> found. Think time: $thinkTime")
            }
            thinkingStringBuilder.append("\n")
        }
    }

    // ===================================================================
    //  Public Getters
    // ===================================================================

    fun getRawResult(): String = rawStringBuilder.toString()

    fun getThinkingContent(): String {
        return if (currentFormat == StreamFormat.THINK_TAGS) {
            if (thinkHasContent) thinkingStringBuilder.toString() else ""
        } else {
            thinkingStringBuilder.toString()
        }
    }

    fun getNormalOutput(): String = normalStringBuilder.toString()

    fun getDisplayResult(): String = getThinkingContent() + getNormalOutput()
}

