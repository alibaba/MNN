// Created by ruoyi.sjd on 2025/2/13.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.utils

import android.content.Context

object ModelPreferences {
    const val TAG: String = "ModelPreferences"

    const val KEY_USE_MMAP: String = "USE_MMAP"


    fun setBoolean(context: Context, modelId: String, key: String?, value: Boolean) {
        context.getSharedPreferences(ModelUtils.safeModelId(modelId), Context.MODE_PRIVATE)
            .edit().putBoolean(key, value).apply()
    }

    @JvmStatic
    fun useMmap(context: Context, modelId: String): Boolean {
        return getBoolean(context, modelId, KEY_USE_MMAP, true)
    }

    fun getBoolean(
        context: Context,
        modelId: String,
        key: String?,
        defaultValue: Boolean
    ): Boolean {
        return context.getSharedPreferences(ModelUtils.safeModelId(modelId), Context.MODE_PRIVATE)
            .getBoolean(key, defaultValue)
    }
}
