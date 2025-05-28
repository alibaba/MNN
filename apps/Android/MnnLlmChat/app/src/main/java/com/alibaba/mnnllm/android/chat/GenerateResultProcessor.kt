// Created by ruoyi.sjd on 2025/2/6.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.chat
import android.util.Log
import java.lang.StringBuilder

class GenerateResultProcessor(
) {

    private var generateBeginTime: Long = 0
    private var isThinking: Boolean = false
    private var hasThought: Boolean = false
    private var thinkHasContent = false

    // Stores the raw input stream
    private val rawStringBuilder = StringBuilder()
    // Stores content within <think> tags, including prefixes
    val thinkingStringBuilder = StringBuilder()
    // Stores content outside <think> tags (final output)
    val normalStringBuilder = StringBuilder()
    var thinkTime = -1L

    // Buffer for handling tags potentially split across data chunks
    private var tagBuffer = ""

    init {
        // Initialization if needed
    }

    /**
     * Call this when the generation process begins to set the start time.
     */
    fun generateBegin() {
        this.generateBeginTime = System.currentTimeMillis()
        Log.d(TAG, "generateBegin ${this.generateBeginTime }")
    }

    /**
     * Gets the raw, unprocessed input string received so far.
     */
    fun getRawResult(): String {
        return rawStringBuilder.toString()
    }

    /**
     * Gets the content generated within the <think> blocks.
     */
    fun getThinkingContent(): String {
        return if (thinkHasContent) thinkingStringBuilder.toString() else ""
    }

    /**
     * Gets the final output content (outside <think> blocks).
     */
    fun getNormalOutput(): String {
        return normalStringBuilder.toString()
    }

    /**
     * Processes a chunk of the LLM output stream.
     *
     * @param progress The incoming string chunk, or null if the stream has ended.
     */
    fun process(progress: String?) {
        Log.d(TAG, "process: #${progress}# thinkingStringBuilder ${this.thinkingStringBuilder}")
        if (progress == null) {
            // Handle end of stream: if anything is left in tagBuffer, treat it as normal text.
            if (tagBuffer.isNotEmpty()) {
                handleText(tagBuffer)
                tagBuffer = ""
            }
            // If thinking was in progress and never ended, add complete prefix.
            // Consider if this is the desired behavior or if an unclosed <think> is an error.
            if (isThinking && hasThought) {
                handleThinkEnd(force = true) // Force end on stream completion
            }
            return
        }

        rawStringBuilder.append(progress)
        var buffer = tagBuffer + progress
        tagBuffer = ""

        while (buffer.isNotEmpty()) {
            val thinkTag = "<think>"
            val endThinkTag = "</think>"

            val thinkIndex = buffer.indexOf(thinkTag)
            val endThinkIndex = buffer.indexOf(endThinkTag)

            when {
                // Case 1: Currently inside a <think> block
                isThinking -> {
                    if (endThinkIndex != -1) {
                        // Found </think>. Process text before it, then handle the tag.
                        val text = buffer.substring(0, endThinkIndex)
                        handleText(text)
                        handleThinkEnd()
                        buffer = buffer.substring(endThinkIndex + endThinkTag.length)
                    } else {
                        // No </think> found. Check for incomplete tag at the end.
                        if (endThinkTag.startsWith(buffer) && buffer.length < endThinkTag.length) {
                            tagBuffer = buffer
                            buffer = ""
                        } else {
                            // Process the whole buffer as thinking text.
                            handleText(buffer)
                            buffer = ""
                        }
                    }
                }
                // Case 2: Currently outside a <think> block
                else -> {
                    if (thinkIndex != -1) {
                        // Found <think>. Process text before it, then handle the tag.
                        val text = buffer.substring(0, thinkIndex)
                        handleText(text)
                        handleThinkStart()
                        buffer = buffer.substring(thinkIndex + thinkTag.length)
                    } else {
                        // No <think> found. Check for incomplete tag at the end.
                        if (thinkTag.startsWith(buffer) && buffer.length < thinkTag.length) {
                            tagBuffer = buffer
                            buffer = ""
                        } else {
                            // Process the whole buffer as normal text.
                            handleText(buffer)
                            buffer = ""
                        }
                    }
                }
            }
        }
    }

    /**
     * Handles the start of a <think> block.
     */
    private fun handleThinkStart() {
        isThinking = true
        if (!hasThought) {
            hasThought = true
            thinkingStringBuilder.append("\n> ")
        } else {
            // Handle subsequent <think> blocks if needed, maybe add a separator or newline?
            // For now, just continue adding to the existing builder with the prefix.
            thinkingStringBuilder.append("\n> ") // Add newline/prompt for subsequent thoughts
        }
    }

    /**
     * Handles the end of a <think> block.
     * @param force Indicates if we should end even without an explicit tag (e.g., end of stream).
     */
    private fun handleThinkEnd(force: Boolean = false) {
        isThinking = false
        thinkTime = (System.currentTimeMillis() - generateBeginTime)
        thinkingStringBuilder.append("\n")
        Log.d(TAG, "handleThinkEnd thinkTime ${this.thinkTime }")
    }

    /**
     * Handles a piece of text, directing it to the correct StringBuilder.
     *
     * @param text The text to handle.
     */
    private fun handleText(text: String) {
        if (text.isEmpty()) return
        if (isThinking) {
            thinkHasContent = thinkHasContent || text.isNotBlank()
            thinkingStringBuilder.append(text.replace("\n", "\n> "))
        } else {
            normalStringBuilder.append(text)
        }
    }

    fun getDisplayResult(): String {
        return thinkingStringBuilder.toString() + normalStringBuilder.toString()
    }

    companion object {
        const val TAG: String = "GenerateResultProcessor"
    }
}