// Created by mnnchat-issue-killer on 2026/03/09.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.chat.chatlist

import androidx.annotation.Nullable
import io.noties.markwon.ext.latex.JLatexMathNode
import org.commonmark.node.Node
import java.util.regex.Pattern
import io.noties.markwon.inlineparser.InlineProcessor

/**
 * Custom inline processor for LaTeX formulas with single dollar sign ($...$) format.
 * This complements the default JLatexMathInlineProcessor which only handles $$...$$ format.
 */
class LatexInlineProcessor : InlineProcessor() {

    companion object {
        // Match single dollar sign format: $...$
        // We use a negative lookbehind/lookahead to avoid matching $$...$$
        // In streaming mode, the closing $ might be missing, but Markwon's InlineProcessor
        // uses regex find, so we must match closed ones here.
        // The unclosed handling will be done at the ChatViewHolders level before parsing.
        private val RE = Pattern.compile("(?<!\\$)\\$(?!\\$)([\\s\\S]+?)(?<!\\$)\\$(?!\\$)")
    }

    override fun specialCharacter(): Char {
        return '$'
    }

    @Nullable
    override
    fun parse(): Node? {
        val match = match(RE) ?: return null
        
        // Extract content between single dollar signs
        // The regex captures the content in group 1
        val content = match.substring(1, match.length - 1)
        
        val node = JLatexMathNode()
        node.latex(content)
        return node
    }
}
