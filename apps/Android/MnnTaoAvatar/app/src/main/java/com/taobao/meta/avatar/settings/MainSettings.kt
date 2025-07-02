// Created by ruoyi.sjd on 2025/3/26.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.taobao.meta.avatar.settings

import android.content.Context
import android.preference.PreferenceManager
import com.taobao.meta.avatar.R


object MainSettings {

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

}