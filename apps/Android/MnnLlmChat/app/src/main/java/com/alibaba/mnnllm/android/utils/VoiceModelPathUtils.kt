// Created by ruoyi.sjd on 2025/01/17.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.utils

import android.content.Context
import android.util.Log
import com.alibaba.mls.api.download.ModelDownloadManager
import com.alibaba.mnnllm.android.mainsettings.MainSettings
import java.io.File

/**
 * Utility class for getting voice model paths dynamically based on MainSettings defaults
 */
object VoiceModelPathUtils {
    private const val TAG = "VoiceModelPathUtils"
    
    // Fallback paths when no default model is set
    private const val FALLBACK_ASR_MODEL_DIR = "/data/local/tmp/asr_models"
    private const val FALLBACK_TTS_MODEL_DIR = "/data/local/tmp/test_new_tts/bert-vits/"
    
    /**
     * Get the ASR model directory path based on the default ASR model setting
     * @param context Android context
     * @return ASR model directory path, or fallback path if no default model is set
     */
    fun getAsrModelPath(context: Context): String {
        val defaultAsrModel = MainSettings.getDefaultAsrModel(context)
        Log.d(TAG, "Getting ASR model path for default model: $defaultAsrModel")
        
        if (defaultAsrModel.isNullOrEmpty()) {
            Log.w(TAG, "No default ASR model set, using fallback path: $FALLBACK_ASR_MODEL_DIR")
            return FALLBACK_ASR_MODEL_DIR
        }
        
        try {
            val modelDownloadManager = ModelDownloadManager.getInstance(context)
            val asrModelFile = modelDownloadManager.getDownloadedFile(defaultAsrModel)
            
            if (asrModelFile != null && asrModelFile.exists()) {
                val modelPath = asrModelFile.absolutePath
                Log.i(TAG, "Found ASR model path: $modelPath for model: $defaultAsrModel")
                return modelPath
            } else {
                Log.w(TAG, "ASR model file not found for: $defaultAsrModel, using fallback path: $FALLBACK_ASR_MODEL_DIR")
                return FALLBACK_ASR_MODEL_DIR
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error getting ASR model path for: $defaultAsrModel", e)
            return FALLBACK_ASR_MODEL_DIR
        }
    }
    
    /**
     * Get the TTS model directory path based on the default TTS model setting
     * @param context Android context
     * @return TTS model directory path, or fallback path if no default model is set
     */
    fun getTtsModelPath(context: Context): String {
        val defaultTtsModel = MainSettings.getDefaultTtsModel(context)
        Log.d(TAG, "Getting TTS model path for default model: $defaultTtsModel")
        
        if (defaultTtsModel.isNullOrEmpty()) {
            Log.w(TAG, "No default TTS model set, using fallback path: $FALLBACK_TTS_MODEL_DIR")
            return FALLBACK_TTS_MODEL_DIR
        }
        
        try {
            val modelDownloadManager = ModelDownloadManager.getInstance(context)
            val ttsModelFile = modelDownloadManager.getDownloadedFile(defaultTtsModel)
            
            if (ttsModelFile != null && ttsModelFile.exists()) {
                val modelPath = ttsModelFile.absolutePath
                Log.i(TAG, "Found TTS model path: $modelPath for model: $defaultTtsModel")
                return modelPath
            } else {
                Log.w(TAG, "TTS model file not found for: $defaultTtsModel, using fallback path: $FALLBACK_TTS_MODEL_DIR")
                return FALLBACK_TTS_MODEL_DIR
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error getting TTS model path for: $defaultTtsModel", e)
            return FALLBACK_TTS_MODEL_DIR
        }
    }
    
    /**
     * Check if voice models are ready and get their status
     * @param context Android context
     * @return Pair<Boolean, String> where first is readiness status and second is status message
     */
    fun checkVoiceModelsStatus(context: Context): Pair<Boolean, String> {
        val defaultTtsModel = MainSettings.getDefaultTtsModel(context)
        val defaultAsrModel = MainSettings.getDefaultAsrModel(context)
        
        val statusBuilder = StringBuilder()
        
        if (defaultTtsModel.isNullOrEmpty()) {
            statusBuilder.append("No default TTS model set. ")
        }
        if (defaultAsrModel.isNullOrEmpty()) {
            statusBuilder.append("No default ASR model set. ")
        }
        
        if (statusBuilder.isNotEmpty()) {
            Log.w(TAG, "Voice models not ready: $statusBuilder")
            return false to statusBuilder.toString()
        }
        
        try {
            val modelDownloadManager = ModelDownloadManager.getInstance(context)
            
            val ttsModelFile = modelDownloadManager.getDownloadedFile(defaultTtsModel!!)
            val asrModelFile = modelDownloadManager.getDownloadedFile(defaultAsrModel!!)
            
            if (ttsModelFile == null || !ttsModelFile.exists()) {
                statusBuilder.append("TTS model file not found: $defaultTtsModel. ")
            }
            if (asrModelFile == null || !asrModelFile.exists()) {
                statusBuilder.append("ASR model file not found: $defaultAsrModel. ")
            }
            
            if (statusBuilder.isNotEmpty()) {
                Log.w(TAG, "Voice models not ready: $statusBuilder")
                return false to statusBuilder.toString()
            }
            
            Log.i(TAG, "Voice models ready - TTS: $defaultTtsModel, ASR: $defaultAsrModel")
            return true to "Voice models ready"
            
        } catch (e: Exception) {
            val errorMsg = "Error checking voice models: ${e.message}"
            Log.e(TAG, errorMsg, e)
            return false to errorMsg
        }
    }
} 