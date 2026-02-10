// Created by ruoyi.sjd on 2025/1/9.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.chat.chatlist

import android.content.Context
import android.net.Uri
import android.util.Log
import android.widget.ImageView
import android.widget.Toast
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.widgets.FullScreenVideoPlayer
import com.alibaba.mnnllm.android.chat.model.ChatDataItem

class VideoPlayerComponent(private val chatDataItem: ChatDataItem) {
    
    fun playVideo(context: Context) {
        val videoUri = chatDataItem.videoUri
        Log.d(TAG, "playVideo called with videoUri: $videoUri")
        if (videoUri != null) {
            try {
                Log.d(TAG, "Starting FullScreenVideoPlayer")
                FullScreenVideoPlayer.showVideoPlayer(context, videoUri)
            } catch (e: Exception) {
                Log.e(TAG, "Error playing video", e)
                Toast.makeText(context, "Error playing video", Toast.LENGTH_SHORT).show()
            }
        } else {
            Log.w(TAG, "No video available")
            Toast.makeText(context, "No video available", Toast.LENGTH_SHORT).show()
        }
    }

    companion object {
        private const val TAG = "VideoPlayerComponent"
    }
}
