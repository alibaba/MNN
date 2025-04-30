// Created by ruoyi.sjd on 2025/2/6.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.chat

import android.util.Log
import okhttp3.RequestBody.Companion.toRequestBody

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
            var currentProgress = progress
            if (currentProgress == null) {
                processEnded = true
                return
            }
            rawStringBuilder.append(currentProgress)
            val currentTokenIndex = nextTokenIndex
            nextTokenIndex++
            //process first token
            if (currentTokenIndex == 0 && currentProgress == "<think>") {
                Log.d(TAG, "thinkingStarted")
                thinkingStarted = true //think mode
            } else if (!thinkingStarted) {//non thinking mode
                Log.d(TAG, "!thinkingStarted")
                isThinking = false
                displayStringBuilder.append(currentProgress)
            } else if ("</think>" == currentProgress) {
                thinkingEnded = true
                if (isThinking == null) {
                    isThinking = false
                }
                if (thinkingStarted) {
                    val thinkTime = (System.currentTimeMillis() - this.generateBeginTime) / 1000
                    displayStringBuilder.replace(
                        0, thinkingPrefix.length,
                        thinkCompletePrefix.replace("ss", thinkTime.toString())
                    )
                }
            } else if (thinkingEnded) {//after thinking
                displayStringBuilder.append(currentProgress)
            } else {//in thinking
                if (isThinking == null && currentProgress.isNotBlank()) {
                    Log.d(TAG, "make sure it is thinking")
                    //make sure it is thinking, if all blank it is false thinking
                    isThinking = true
                    displayStringBuilder.append(thinkingPrefix).append("\n")
                    displayStringBuilder.append(pendingBlanks.toString())
                    displayStringBuilder.append("> ")
                }
                if (isThinking == true) {
                    Log.d(TAG, "append thinking ")
                    if (currentProgress.contains("\n")) {
                        currentProgress = currentProgress.replace("\n", "\n> ")
                    }
                    displayStringBuilder.append(currentProgress)
                } else {
                    Log.d(TAG, "pending thinking spaced ")
                    pendingBlanks.append(currentProgress)
                }
            }
        }

        companion object {
            const val TAG: String = "R1GenerateResultProcessor"
        }
    }

}
