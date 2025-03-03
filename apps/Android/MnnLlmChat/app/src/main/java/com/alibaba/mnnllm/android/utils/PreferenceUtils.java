// Created by ruoyi.sjd on 2025/1/13.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.utils;

import android.content.Context;
import android.preference.PreferenceManager;

public class PreferenceUtils {

    public static final String TAG = "PreferenceUtils";
    public static final String KEY_SHOW_PERFORMACE_METRICS = "SHOW_PERFORMACE_METRICS";

    public static final String KEY_USE_MODELSCOPE_DOWNLOAD = "USE_MODELSCOPE_DOWNLOAD";

    public static final String KEY_LIST_FILTER_ONLY_DOWNLOADED = "LIST_FILTER_ONLY_DOWNLOADED";


    public static void setBoolean(Context context, String key , boolean value) {
        PreferenceManager.getDefaultSharedPreferences(context).edit().putBoolean(key, value).apply();
    }

    public static void setLong(Context context, String key , long value) {
        PreferenceManager.getDefaultSharedPreferences(context).edit().putLong(key, value).apply();
    }

    public static long getLong(Context context, String key , long defaultValue) {
        return PreferenceManager.getDefaultSharedPreferences(context).getLong(key, defaultValue);
    }

    public static boolean getBoolean(Context context, String key , boolean defaultValue) {
        return PreferenceManager.getDefaultSharedPreferences(context).getBoolean(key, defaultValue);
    }



    public static boolean isUseModelsScopeDownload(Context context) {
        boolean defaultValue = DeviceUtils.isChinese();
        if (!PreferenceManager.getDefaultSharedPreferences(context).getAll().containsKey(KEY_USE_MODELSCOPE_DOWNLOAD)) {
            setUseModelsScopeDownload(context, defaultValue);
            return defaultValue;
        }
        return getBoolean(context, KEY_USE_MODELSCOPE_DOWNLOAD, defaultValue);
    }

    public static void setUseModelsScopeDownload(Context context, boolean value) {
        setBoolean(context, KEY_USE_MODELSCOPE_DOWNLOAD, value);
    }

    public static void setFilterDownloaded(Context context, boolean filterDownloaded) {
        setBoolean(context, KEY_LIST_FILTER_ONLY_DOWNLOADED, filterDownloaded);
    }

    public static boolean isFilterDownloaded(Context context) {
        return getBoolean(context, KEY_LIST_FILTER_ONLY_DOWNLOADED, false);
    }
}
