// Created by ruoyi.sjd on 2025/01/17.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.chat.voice

import android.content.Context
import android.util.Log
import com.alibaba.mls.api.download.ModelDownloadManager
import com.alibaba.mnnllm.android.mainsettings.MainSettings

/**
 * Responsible for checking if voice chat models are ready.
 * Only checks if default TTS and ASR models are set and their paths exist.
 */
class VoiceModelsChecker(private val context: Context) {

    private val modelDownloadManager = ModelDownloadManager.getInstance(context)

    /**
     * Check if voice chat is ready by verifying:
     * 1. Default TTS model is set and exists
     * 2. Default ASR model is set and exists
     */
    fun isVoiceChatReady(): Boolean {
        val defaultTtsModel = MainSettings.getDefaultTtsModel(context)
        val defaultAsrModel = MainSettings.getDefaultAsrModel(context)
        
        Log.d(TAG, "Checking voice chat readiness - TTS: $defaultTtsModel, ASR: $defaultAsrModel")
        
        // Check if both models are set
        if (defaultTtsModel.isNullOrEmpty() || defaultAsrModel.isNullOrEmpty()) {
            Log.d(TAG, "Voice chat not ready: models not set")
            return false
        }
        
        // Check if TTS model file exists
        val ttsModelFile = modelDownloadManager.getDownloadedFile(defaultTtsModel)
        if (ttsModelFile == null || !ttsModelFile.exists()) {
            Log.d(TAG, "Voice chat not ready: TTS model file not found - $defaultTtsModel")
            return false
        }
        
        // Check if ASR model file exists
        val asrModelFile = modelDownloadManager.getDownloadedFile(defaultAsrModel)
        if (asrModelFile == null || !asrModelFile.exists()) {
            Log.d(TAG, "Voice chat not ready: ASR model file not found - $defaultAsrModel")
            return false
        }
        
        Log.d(TAG, "Voice chat ready: both TTS and ASR models are available")
        return true
    }
    
    /**
     * Get information about what's missing for voice chat
     */
    fun getVoiceChatStatus(): VoiceChatStatus {
        val defaultTtsModel = MainSettings.getDefaultTtsModel(context)
        val defaultAsrModel = MainSettings.getDefaultAsrModel(context)
        
        val ttsModelSet = !defaultTtsModel.isNullOrEmpty()
        val asrModelSet = !defaultAsrModel.isNullOrEmpty()
        
        val ttsModelExists = if (ttsModelSet) {
            val ttsModelFile = modelDownloadManager.getDownloadedFile(defaultTtsModel!!)
            ttsModelFile != null && ttsModelFile.exists()
        } else {
            false
        }
        
        val asrModelExists = if (asrModelSet) {
            val asrModelFile = modelDownloadManager.getDownloadedFile(defaultAsrModel!!)
            asrModelFile != null && asrModelFile.exists()
        } else {
            false
        }
        
        return VoiceChatStatus(
            ttsModelSet = ttsModelSet,
            asrModelSet = asrModelSet,
            ttsModelExists = ttsModelExists,
            asrModelExists = asrModelExists,
            defaultTtsModel = defaultTtsModel,
            defaultAsrModel = defaultAsrModel
        )
    }

    companion object {
        const val TAG = "VoiceModelsChecker"
    }
}

/**
 * Data class to hold voice chat status information
 */
data class VoiceChatStatus(
    val ttsModelSet: Boolean,
    val asrModelSet: Boolean,
    val ttsModelExists: Boolean,
    val asrModelExists: Boolean,
    val defaultTtsModel: String?,
    val defaultAsrModel: String?
) {
    val isReady: Boolean
        get() = ttsModelSet && asrModelSet && ttsModelExists && asrModelExists
        
    val hasRequiredModels: Boolean
        get() = ttsModelExists && asrModelExists
        
    val needsModelSelection: Boolean
        get() = !ttsModelSet || !asrModelSet
        
    val needsModelDownload: Boolean
        get() = (ttsModelSet && !ttsModelExists) || (asrModelSet && !asrModelExists)
} 