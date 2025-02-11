// Created by ruoyi.sjd on 2025/1/13.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.utils;

import android.content.Context;
import android.preference.PreferenceManager;

public class PreferenceUtils {

    public static final String TAG = "PreferenceUtils";
    public static final String KEY_SHOW_PERFORMACE_METRICS = "SHOW_PERFORMACE_METRICS";

    public static final String KEY_USE_MODELSCOPE_DOWNLOAD = "USE_MODELSCOPE_DOWNLOAD";

    public static void setBoolean(Context context, String key , boolean value) {
        PreferenceManager.getDefaultSharedPreferences(context).edit().putBoolean(key, value).apply();
    }

    public static boolean getBoolean(Context context, String key , boolean defaultValue) {
        return PreferenceManager.getDefaultSharedPreferences(context).getBoolean(key, defaultValue);
    }

    public static boolean isUseModelsScopeDownload(Context context) {
        return getBoolean(context, KEY_USE_MODELSCOPE_DOWNLOAD, false);
    }

    public static void setUseModelsScopeDownload(Context context, boolean value) {
        setBoolean(context, KEY_USE_MODELSCOPE_DOWNLOAD, value);
    }
}
