// Created by ruoyi.sjd on 2025/1/9.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.widgets

import android.app.Activity
import android.content.Context
import android.net.Uri
import android.os.Bundle
import android.view.View
import android.widget.MediaController
import android.widget.VideoView
import androidx.appcompat.app.AppCompatActivity
import com.alibaba.mnnllm.android.R

class FullScreenVideoPlayer : AppCompatActivity() {
    
    private lateinit var videoView: VideoView
    private var videoUri: Uri? = null
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_fullscreen_video_player)
        
        videoUri = intent.getParcelableExtra(EXTRA_VIDEO_URI)
        
        videoView = findViewById(R.id.video_view)
        setupVideoPlayer()
    }
    
    private fun setupVideoPlayer() {
        videoUri?.let { uri ->
            videoView.setVideoURI(uri)
            
            val mediaController = MediaController(this)
            mediaController.setAnchorView(videoView)
            videoView.setMediaController(mediaController)
            
            videoView.setOnPreparedListener { mp ->
                mp.start()
            }
            
            videoView.setOnCompletionListener { mp ->
                mp.start()
            }
        }
    }
    
    override fun onPause() {
        super.onPause()
        videoView.pause()
    }
    
    override fun onDestroy() {
        super.onDestroy()
        videoView.stopPlayback()
    }
    
    companion object {
        private const val EXTRA_VIDEO_URI = "extra_video_uri"
        
        fun showVideoPlayer(context: Context, videoUri: Uri) {
            val intent = android.content.Intent(context, FullScreenVideoPlayer::class.java)
            intent.putExtra(EXTRA_VIDEO_URI, videoUri)
            if (context is Activity) {
                context.startActivity(intent)
            } else {
                intent.addFlags(android.content.Intent.FLAG_ACTIVITY_NEW_TASK)
                context.startActivity(intent)
            }
        }
    }
}
