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
        } else if (!userData.imageUris.isNullOrEmpty()) {
            val sb = StringBuilder()
            for (uri in userData.imageUris!!) {
                val imagePath = FileUtils.getPathForUri(uri)
                    ?: throw Exception("imagePath path is null")
                sb.append(String.format("<img>%s</img>", imagePath))
            }
            sb.append(userData.text ?: "")
            input = sb.toString()
            android.util.Log.d("PromptUtils", "Generated input with images: $input")
        } else if (userData.videoUri != null) {
            val videoPath = FileUtils.getPathForUri(userData.videoUri!!)
                ?: throw Exception("videoPath path is null")
            input = String.format("<video>%s</video>%s", videoPath, userData.text)
        } else {
            input = userData.text!!
        }
        return input
    }
}