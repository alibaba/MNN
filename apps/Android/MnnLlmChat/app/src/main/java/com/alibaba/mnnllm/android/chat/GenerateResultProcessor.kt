// Created by ruoyi.sjd on 2025/2/6.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.chat

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
        private val pendingBlanks = StringBuilder()
        private var processEnded = false
        private var isThinking:Boolean? = null
        private var thinkingStarted = false
        private var thinkingEnded = false
        private var firstToken:String? = null
        private var nextTokenIndex = 0
        private var tagBuffer = ""

        init {
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
            if (progress == null) {
                processEnded = true
                return
            }
            rawStringBuilder.append(progress)
            var buffer = tagBuffer + progress
            tagBuffer = ""

            while (buffer.isNotEmpty()) {
                when {
                    buffer.startsWith("<think>") -> {
                        handleThinkStart()
                        buffer = buffer.removePrefix("<think>")
                    }
                    buffer.startsWith("</think>") -> {
                        if (nextTokenIndex == 0) {
                        } else {
                            handleThinkEnd()
                        }
                        buffer = buffer.removePrefix("</think>")
                    }
                    buffer.startsWith("<") && !buffer.contains(">") -> {
                        tagBuffer = buffer
                        break
                    }
                    else -> {
                        val nextTag = buffer.indexOf('<')
                        val text = if (nextTag == -1) buffer else buffer.substring(0, nextTag)
                        handleText(text)
                        buffer = if (nextTag == -1) "" else buffer.substring(nextTag)
                    }
                }
            }
        }

        private fun handleThinkStart() {
            if (nextTokenIndex == 0) {
                thinkingStarted = true
                isThinking = true
                displayStringBuilder.append(thinkingPrefix).append("\n> ")
            }
            nextTokenIndex++
        }

        private fun handleThinkEnd() {
            thinkingEnded = true
            if (isThinking == null) isThinking = false
            if (thinkingStarted) {
                val thinkTime = (System.currentTimeMillis() - generateBeginTime) / 1000
                displayStringBuilder.replace(
                    0, thinkingPrefix.length,
                    thinkCompletePrefix.replace("ss", thinkTime.toString())
                )
            }
            nextTokenIndex++
        }

        private fun handleText(text: String) {
            if (!thinkingStarted) {
                isThinking = false
                displayStringBuilder.append(text)
            } else if (thinkingEnded) {
                displayStringBuilder.append(text)
            } else {
                if (isThinking == null && text.isNotBlank()) {
                    isThinking = true
                    displayStringBuilder.append(thinkingPrefix).append("\n> ")
                }
                if (isThinking == true) {
                    displayStringBuilder.append(text.replace("\n", "\n> "))
                } else {
                    pendingBlanks.append(text)
                }
            }
        }

        companion object {
            const val TAG: String = "R1GenerateResultProcessor"
        }
    }

}
