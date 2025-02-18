// Created by ruoyi.sjd on 2025/2/13.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.utils;

import static com.alibaba.mnnllm.android.utils.ModelUtils.safeModelId;

import android.content.Context;

public class ModelPreferences {

    public static final String TAG = "ModelPreferences";

    public static final String KEY_USE_MMAP = "USE_MMAP";


    public static void setBoolean(Context context,String modelId, String key , boolean value) {
        context.getSharedPreferences(safeModelId(modelId), Context.MODE_PRIVATE)
                .edit().putBoolean(key, value).apply();
    }

    public static boolean useMmap(Context context, String modelId) {
        return getBoolean(context, modelId, KEY_USE_MMAP, true);
    }
    public static boolean getBoolean(Context context, String modelId, String key , boolean defaultValue) {
        return context.getSharedPreferences(safeModelId(modelId), Context.MODE_PRIVATE)
                .getBoolean(key, defaultValue);
    }
}
