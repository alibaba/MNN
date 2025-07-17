// Created by ruoyi.sjd on 2025/3/19.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.taobao.meta.avatar.llm

import android.annotation.SuppressLint
import android.graphics.Color
import android.text.Spannable
import android.text.SpannableString
import android.text.SpannableStringBuilder
import android.text.style.ForegroundColorSpan
import android.widget.TextView
import kotlinx.coroutines.MainScope
import kotlinx.coroutines.launch

class LlmPresenter(private val textResponse: TextView) {

    private var builder = SpannableStringBuilder()
    private val MAX_LENGTH = 10000
    private var lastMessageRole: String? = null
    private var lastMessageStartIndex: Int = 0
    private var currentAIMessage = ""
    private var stopped = false
    private var currentSessionId = 0L

    fun reset() {
        stop()
        currentSessionId = 0L
        textResponse.text = ""
    }

    fun setCurrentSessionId(sessionId: Long) {
        currentSessionId = sessionId
    }

    fun stop() {
        stopped = true
    }

    fun start() {
        stopped = false
    }

    @SuppressLint("SetTextI18n")
    fun onLlmTextUpdate(totalText: String, callingSessionId: Long) {
        MainScope().launch {
            if (callingSessionId != currentSessionId) {
                return@launch
            }
            addMessage("ai", totalText)
        }
    }

    fun onUserTextUpdate(text: String) {
        MainScope().launch {
            addMessage("human", text)
        }
    }

    @SuppressLint("ClickableViewAccessibility")
    fun addMessage(role: String, message: String) {
        val lowerRole = role.lowercase()
        val color = when (lowerRole) {
            "ai" -> Color.BLACK
            "human" -> Color.GRAY
            else -> Color.BLACK
        }

        if (lowerRole == "ai") {
            if (lastMessageRole == "ai") {
                currentAIMessage += message
                builder.delete(lastMessageStartIndex, builder.length)
            } else {
                currentAIMessage = message
                lastMessageStartIndex = builder.length
            }
            val updatedMessage = currentAIMessage + "\n"
            val spannable = SpannableString(updatedMessage)
            spannable.setSpan(
                ForegroundColorSpan(color),
                0,
                spannable.length,
                Spannable.SPAN_EXCLUSIVE_EXCLUSIVE
            )
            builder.append(spannable)
        } else {
            currentAIMessage = ""
            lastMessageStartIndex = builder.length
            val formattedMessage = message + "\n"
            val spannable = SpannableString(formattedMessage)
            spannable.setSpan(
                ForegroundColorSpan(color),
                0,
                spannable.length,
                Spannable.SPAN_EXCLUSIVE_EXCLUSIVE
            )
            builder.append(spannable)
        }
        lastMessageRole = lowerRole
        trimIfNeeded()
        textResponse.text = builder
        onScrollToBottom()
    }

    private fun trimIfNeeded() {
        while (builder.length > MAX_LENGTH) {
            val firstNewline = builder.indexOf("\n")
            if (firstNewline != -1) {
                builder.delete(0, firstNewline + 1)
            } else {
                builder.clear()
            }
        }
    }

    private fun onScrollToBottom() {
        if (textResponse.visibility != TextView.VISIBLE || textResponse.layout == null) {
            return
        }
        val scrollAmount =
            textResponse.layout.getLineTop(textResponse!!.lineCount) - textResponse!!.height
        if (scrollAmount > 0) {
            textResponse.scrollTo(
                0,
                scrollAmount + 100
            )
        } else {
            textResponse.scrollTo(0, 0)
        }
    }

    fun onEndCall() {
        builder = SpannableStringBuilder()
        textResponse.text = ""
        lastMessageRole = null
        lastMessageStartIndex = 0
        currentAIMessage = ""
    }

}