// Created by ruoyi.sjd on 2025/1/9.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.widgets

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import android.util.AttributeSet
import android.view.LayoutInflater
import android.widget.FrameLayout
import android.widget.ImageView
import androidx.core.content.ContextCompat
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.utils.VideoThumbnailUtils
import android.view.MotionEvent

class VideoPreviewView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : FrameLayout(context, attrs, defStyleAttr) {
    
    private val thumbnailImageView: ImageView
    private val playIconImageView: ImageView
    
    init {
        LayoutInflater.from(context).inflate(R.layout.view_video_preview, this, true)
        thumbnailImageView = findViewById(R.id.video_thumbnail)
        playIconImageView = findViewById(R.id.video_play_icon)
        
        // Ensure the view is clickable
        isClickable = true
        isFocusable = true
        
        // Make child views non-clickable so they don't intercept touch events
        thumbnailImageView.isClickable = false
        thumbnailImageView.isFocusable = false
        playIconImageView.isClickable = false
        playIconImageView.isFocusable = false
    }
    
    /**
     * Set video URI and generate thumbnail
     */
    fun setVideoUri(videoUri: Uri?) {
        android.util.Log.d("VideoPreviewView", "setVideoUri called with: $videoUri")
        if (videoUri == null) {
            android.util.Log.d("VideoPreviewView", "videoUri is null, setting default icon")
            thumbnailImageView.setImageResource(R.drawable.ic_video)
            return
        }
        
        // Asynchronously generate video thumbnail
        Thread {
            try {
                val thumbnail = VideoThumbnailUtils.generateVideoThumbnail(context, videoUri, 200, 200)
                post {
                    if (thumbnail != null) {
                        thumbnailImageView.setImageBitmap(thumbnail)
                    } else {
                        // If thumbnail generation fails, show default video icon
                        thumbnailImageView.setImageResource(R.drawable.ic_video)
                    }
                }
            } catch (e: Exception) {
                post {
                    thumbnailImageView.setImageResource(R.drawable.ic_video)
                }
            }
        }.start()
    }
    
    /**
     * Set video path and generate thumbnail
     */
    fun setVideoPath(videoPath: String?) {
        if (videoPath.isNullOrEmpty()) {
            thumbnailImageView.setImageResource(R.drawable.ic_video)
            return
        }
        
        // Asynchronously generate video thumbnail
        Thread {
            try {
                val thumbnail = VideoThumbnailUtils.generateVideoThumbnailFromPath(videoPath, 200, 200)
                post {
                    if (thumbnail != null) {
                        thumbnailImageView.setImageBitmap(thumbnail)
                    } else {
                        // If thumbnail generation fails, show default video icon
                        thumbnailImageView.setImageResource(R.drawable.ic_video)
                    }
                }
            } catch (e: Exception) {
                post {
                    thumbnailImageView.setImageResource(R.drawable.ic_video)
                }
            }
        }.start()
    }
    
    /**
     * Set thumbnail
     */
    fun setThumbnail(bitmap: Bitmap?) {
        if (bitmap != null) {
            thumbnailImageView.setImageBitmap(bitmap)
        } else {
            thumbnailImageView.setImageResource(R.drawable.ic_video)
        }
    }
    
    /**
     * Set play icon visibility
     */
    fun setPlayIconVisible(visible: Boolean) {
        playIconImageView.visibility = if (visible) VISIBLE else GONE
    }

    /**
     * Override onTouchEvent to ensure proper click handling
     */
    override fun onTouchEvent(event: MotionEvent): Boolean {
        android.util.Log.d("VideoPreviewView", "onTouchEvent: ${event.action}")
        when (event.action) {
            MotionEvent.ACTION_DOWN -> {
                // Handle touch down
                android.util.Log.d("VideoPreviewView", "Touch DOWN received")
                return true
            }
            MotionEvent.ACTION_UP -> {
                // Handle touch up (click)
                android.util.Log.d("VideoPreviewView", "Touch UP received, calling performClick")
                if (isClickable) {
                    performClick()
                }
                return true
            }
        }
        return super.onTouchEvent(event)
    }
    
    /**
     * Ensure performClick is properly handled
     */
    override fun performClick(): Boolean {
        android.util.Log.d("VideoPreviewView", "performClick called")
        super.performClick()
        return true
    }
    
    /**
     * Override dispatchTouchEvent to ensure touch events are properly handled
     */
    override fun dispatchTouchEvent(event: MotionEvent): Boolean {
        android.util.Log.d("VideoPreviewView", "dispatchTouchEvent: ${event.action}")
        // First try to handle the touch event ourselves
        if (onTouchEvent(event)) {
            android.util.Log.d("VideoPreviewView", "Touch event handled by onTouchEvent")
            return true
        }
        // If we don't handle it, let the parent handle it
        android.util.Log.d("VideoPreviewView", "Touch event passed to parent")
        return super.dispatchTouchEvent(event)
    }
    
    /**
     * Override onInterceptTouchEvent to ensure child views don't intercept touch events
     */
    override fun onInterceptTouchEvent(ev: MotionEvent?): Boolean {
        android.util.Log.d("VideoPreviewView", "onInterceptTouchEvent: ${ev?.action}")
        // Don't intercept touch events, let them be handled by the parent
        return false
    }
}
