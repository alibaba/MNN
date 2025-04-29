// Created by ruoyi.sjd on 2025/2/6.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.chat

import android.util.Log

interface GenerateResultProcessor {
    fun process(progress: String?)
    fun getDisplayResult(): String
    fun getRawResult(): String
    fun generateBegin()

    open class NormalGenerateResultProcessor : GenerateResultProcessor {
        protected var rawStringBuilder: StringBuilder = StringBuilder()

        override fun process(progress: String?) {
            if (progress != null) {
                rawStringBuilder.append(progress)
            }
        }

        override fun getDisplayResult(): String {
            return rawStringBuilder.toString()
        }

        override fun getRawResult(): String {
            return rawStringBuilder.toString()
        }

        override fun generateBegin() {
        }
    }

    class R1GenerateResultProcessor(thinkingPrefix: String, thinkCompletePrefix: String) :
        NormalGenerateResultProcessor() {
        private val thinkingPrefix: String
        private var generateBeginTime: Long = 0
        private var hasThinkProcessed = false
        private val thinkCompletePrefix: String
        private val displayStringBuilder = StringBuilder()
        private var thinkStarted = false
        private var processEnded = false

        init {
            displayStringBuilder.append(thinkingPrefix).append("\n")
            this.thinkCompletePrefix = thinkCompletePrefix
            this.thinkingPrefix = thinkingPrefix
        }

        override fun getRawResult(): String {
            return super.getRawResult()
        }

        override fun generateBegin() {
            super.generateBegin()
            this.generateBeginTime = System.currentTimeMillis()
        }

        override fun getDisplayResult(): String {
            return displayStringBuilder.toString()
        }

        override fun process(progress: String?) {
            var currentProgress = progress
            if (currentProgress == null) {
                processEnded = true
                return
            }
            if (currentProgress.contains("<think>")) {
                currentProgress = currentProgress.replace("<think>", "")
            }
            rawStringBuilder.append(currentProgress)
            if (currentProgress.contains("</think>")) {
                currentProgress = currentProgress.replace("</think>", "\n")
                val thinkTime = (System.currentTimeMillis() - this.generateBeginTime) / 1000
                displayStringBuilder.replace(
                    0, thinkingPrefix.length,
                    thinkCompletePrefix.replace("ss", thinkTime.toString())
                )
                hasThinkProcessed = true
            } else if (!hasThinkProcessed) {
                if (!thinkStarted) {
                    displayStringBuilder.append("> ")
                    thinkStarted = true
                }
                if (currentProgress.contains("\n")) {
                    currentProgress = currentProgress.replace("\n", "\n> ")
                }
            }
            displayStringBuilder.append(currentProgress)
        }
    }
}
