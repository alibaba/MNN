// Created by ruoyi.sjd on 2025/2/28.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.settings

import android.content.Context
import androidx.preference.PreferenceManager
import com.alibaba.mls.api.source.ModelSources


object MainSettings {

    fun getDownloadProvider(context: Context): ModelSources.ModelSourceType {
        val sharedPreferences = PreferenceManager.getDefaultSharedPreferences(context)
        val result = sharedPreferences.getString("download_provider", "HuggingFace")
        return when (result) {
            "HuggingFace" -> ModelSources.ModelSourceType.HUGGING_FACE
            "ModelScope" -> ModelSources.ModelSourceType.MODEL_SCOPE
            else -> ModelSources.ModelSourceType.MODELERS
         }
    }

    fun isStopDownloadOnChatEnabled(context: Context): Boolean {
        val sharedPreferences = PreferenceManager.getDefaultSharedPreferences(context)
        return sharedPreferences.getBoolean("stop_download_on_chat", true)
    }

}