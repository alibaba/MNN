// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.chat

import android.net.Uri
import android.text.TextUtils
import java.io.File

class ChatDataItem {
    @JvmField
    var time: String? = null
    @JvmField
    var audioPlayComponent: AudioPlayerComponent? = null
    @JvmField
    var text: String? = null
    var type: Int
        private set
    @JvmField
    var imageUri: Uri? = null

    @JvmField
    var audioUri: Uri? = null

    @JvmField
    var benchmarkInfo: String? = null

    var displayText: String? = null
        get() = if (TextUtils.isEmpty(field)) text else field

    var audioDuration = 0f

    constructor(time: String?, type: Int, text: String?) {
        this.time = time
        this.type = type
        this.text = text
    }

    constructor(type: Int) {
        this.type = type
    }

    val audioPath: String?
        get() {
            if (this.audioUri != null && "file" == audioUri!!.scheme) {
                return audioUri!!.path
            }
            return null
        }

    companion object {
        fun createImageInputData(timeString: String?, text: String?, imageUri: Uri?): ChatDataItem {
            val result = ChatDataItem(timeString, ChatViewHolders.USER, text)
            result.imageUri = imageUri
            return result
        }

        fun createAudioInputData(
            timeString: String?,
            text: String?,
            audioPath: String,
            duration: Float
        ): ChatDataItem {
            val result = ChatDataItem(timeString, ChatViewHolders.USER, text)
            result.audioUri = Uri.fromFile(File(audioPath))
            result.audioDuration = duration
            return result
        }
    }
}

