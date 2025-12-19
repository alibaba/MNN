// Created by ruoyi.sjd on 2025/3/26.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.taobao.meta.avatar.settings

import android.content.Context
import android.preference.PreferenceManager
import com.taobao.meta.avatar.R


object MainSettings {
    private const val KEY_TTS_SPEAKER_ID = "tts_speaker_id"
    private const val KEY_TTS_SPEED = "tts_speed"
    private const val DEFAULT_SPEAKER_ID = "F1"

    fun getLlmPrompt(context: Context): String {
        val sharedPreferences = PreferenceManager.getDefaultSharedPreferences(context)
        val llmPrompt = sharedPreferences.getString("llm_prompt", null)
        return if (llmPrompt.isNullOrEmpty()) {
            context.getString(R.string.llm_prompt_default)
        } else {
            llmPrompt
        }
    }

    fun isShowDebugInfo(context: Context): Boolean {
        val sharedPreferences = PreferenceManager.getDefaultSharedPreferences(context)
        return sharedPreferences.getBoolean("show_debug_info", true)
    }
    
    // TTS Speaker ID
    fun getTtsSpeakerId(context: Context): String {
        val sharedPreferences = PreferenceManager.getDefaultSharedPreferences(context)
        return sharedPreferences.getString(KEY_TTS_SPEAKER_ID, DEFAULT_SPEAKER_ID) 
               ?: DEFAULT_SPEAKER_ID
    }
    
    fun setTtsSpeakerId(context: Context, speakerId: String) {
        PreferenceManager.getDefaultSharedPreferences(context)
            .edit()
            .putString(KEY_TTS_SPEAKER_ID, speakerId)
            .apply()
    }
    
    // TTS Speed (范围 0.5 - 2.0，存储时放大10倍为整数 5-20)
    fun getTtsSpeed(context: Context): Float {
        val sharedPreferences = PreferenceManager.getDefaultSharedPreferences(context)
        val speedInt = sharedPreferences.getInt(KEY_TTS_SPEED, 10)
        return speedInt / 10.0f
    }
    
    fun setTtsSpeed(context: Context, speedInt: Int) {
        PreferenceManager.getDefaultSharedPreferences(context)
            .edit()
            .putInt(KEY_TTS_SPEED, speedInt)
            .apply()
    }
}