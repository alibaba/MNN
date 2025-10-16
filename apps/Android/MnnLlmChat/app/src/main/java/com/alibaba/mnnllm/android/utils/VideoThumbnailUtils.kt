// Created by ruoyi.sjd on 2025/1/9.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.utils

import android.content.Context
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.util.Log
import java.io.File

object VideoThumbnailUtils {
    private const val TAG = "VideoThumbnailUtils"
    
    /**
     * Generate video thumbnail
     * @param context Context
     * @param videoUri Video URI
     * @param width Thumbnail width
     * @param height Thumbnail height
     * @return Video thumbnail Bitmap, returns null if failed
     */
    fun generateVideoThumbnail(context: Context, videoUri: Uri, width: Int = 200, height: Int = 200): Bitmap? {
        val mmr = MediaMetadataRetriever()
        return try {
            mmr.setDataSource(context, videoUri)
            // Get the first frame of the video as thumbnail
            val bitmap = mmr.frameAtTime
            if (bitmap != null) {
                // Scale thumbnail to specified dimensions
                Bitmap.createScaledBitmap(bitmap, width, height, true)
            } else {
                null
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to generate video thumbnail", e)
            null
        } finally {
            try {
                mmr.release()
            } catch (e: Exception) {
                Log.e(TAG, "Error releasing MediaMetadataRetriever", e)
            }
        }
    }
    
    /**
     * Generate video thumbnail (from file path)
     * @param videoPath Video file path
     * @param width Thumbnail width
     * @param height Thumbnail height
     * @return Video thumbnail Bitmap, returns null if failed
     */
    fun generateVideoThumbnailFromPath(videoPath: String, width: Int = 200, height: Int = 200): Bitmap? {
        val mmr = MediaMetadataRetriever()
        return try {
            mmr.setDataSource(videoPath)
            // Get the first frame of the video as thumbnail
            val bitmap = mmr.frameAtTime
            if (bitmap != null) {
                // Scale thumbnail to specified dimensions
                Bitmap.createScaledBitmap(bitmap, width, height, true)
            } else {
                null
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to generate video thumbnail from path: $videoPath", e)
            null
        } finally {
            try {
                mmr.release()
            } catch (e: Exception) {
                Log.e(TAG, "Error releasing MediaMetadataRetriever", e)
            }
        }
    }
}
