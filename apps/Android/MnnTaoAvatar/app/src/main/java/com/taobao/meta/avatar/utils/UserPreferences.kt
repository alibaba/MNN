// Created by ruoyi.sjd on 2025/3/20.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.taobao.meta.avatar.utils

import android.content.Context
import android.preference.PreferenceManager

object UserPreferences {
    private const val TAG = "UserPreferences"
    private const val SHOW_LLM_TEXT = "show_llm_text"

    fun isShowLlmText(context: Context): Boolean {
        val sharedPreferences = PreferenceManager.getDefaultSharedPreferences(context)
        return sharedPreferences.getBoolean(SHOW_LLM_TEXT, true)
    }

    fun setShowLlmText(context: Context, showLlmText:Boolean) {
        val sharedPreferences = PreferenceManager.getDefaultSharedPreferences(context)
            .edit()
            .putBoolean(SHOW_LLM_TEXT, showLlmText)
            .apply()
    }

    fun setLong(context: Context?, key: String?, value: Long) {
        PreferenceManager.getDefaultSharedPreferences(context).edit().putLong(key, value).apply()
    }

    fun getLong(context: Context?, key: String?, defaultValue: Long): Long {
        return PreferenceManager.getDefaultSharedPreferences(context).getLong(key, defaultValue)
    }

}