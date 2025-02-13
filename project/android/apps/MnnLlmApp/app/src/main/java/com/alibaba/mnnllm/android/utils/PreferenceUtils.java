// Created by ruoyi.sjd on 2025/1/13.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.utils;

import android.content.Context;
import android.content.res.Configuration;
import android.preference.PreferenceManager;

import com.alibaba.mls.api.ApplicationUtils;

import java.util.Locale;

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

    private static boolean isChinese() {
        Configuration config = ApplicationUtils.get().getResources().getConfiguration();
        Locale locale = config.getLocales().get(0);
        String language = locale.getLanguage();
        String country = locale.getCountry();
        if (language.equals("zh") && country.equals("CN")) {
            return true;
        } else {
            return false;
        }
    }

    public static boolean isUseModelsScopeDownload(Context context) {
        boolean defaultValue = isChinese();
        if (!PreferenceManager.getDefaultSharedPreferences(context).getAll().containsKey(KEY_USE_MODELSCOPE_DOWNLOAD)) {
            setUseModelsScopeDownload(context, defaultValue);
            return defaultValue;
        }
        return getBoolean(context, KEY_USE_MODELSCOPE_DOWNLOAD, defaultValue);
    }

    public static void setUseModelsScopeDownload(Context context, boolean value) {
        setBoolean(context, KEY_USE_MODELSCOPE_DOWNLOAD, value);
    }
}
