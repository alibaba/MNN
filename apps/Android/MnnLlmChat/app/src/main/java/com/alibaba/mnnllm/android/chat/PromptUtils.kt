// Created by ruoyi.sjd on 2025/5/7.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.chat
import com.alibaba.mnnllm.android.chat.model.ChatDataItem
import com.alibaba.mnnllm.android.utils.FileUtils

object PromptUtils {
    fun generateUserPrompt(userData: ChatDataItem): String {
        val input: String
        if (userData.audioUri != null) {
            val audioPath = FileUtils.getPathForUri(userData.audioUri!!)
                ?: throw Exception("Audio path is null")
            if (userData.audioDuration <= 0.1) {
                userData.audioDuration =
                    FileUtils.getAudioDuration(audioPath).toFloat()
            }
            input = String.format("<audio>%s</audio>%s", audioPath, userData.text)
        } else if (userData.imageUri != null) {
            val imagePath = FileUtils.getPathForUri(userData.imageUri!!)
                ?: throw Exception("imagePath path is null")
            input = String.format("<img>%s</img>%s", imagePath, userData.text)
        } else {
            input = userData.text!!
        }
        return input
    }
}