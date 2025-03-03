// Created by ruoyi.sjd on 2025/2/10.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.utils;
import android.content.res.Configuration;
import android.os.Build;

import com.alibaba.mls.api.ApplicationProvider;

import java.util.Locale;

public class DeviceUtils {

    public static String getDeviceInfo() {
        return "DeviceInfo: " + android.os.Build.MANUFACTURER + " "
                + android.os.Build.MODEL + " " + android.os.Build.VERSION.RELEASE +
                "\nSdkInt:" + Build.VERSION.SDK_INT;
    }

    public static boolean isChinese() {
        Configuration config = ApplicationProvider.get().getResources().getConfiguration();
        Locale locale = config.getLocales().get(0);
        String language = locale.getLanguage();
        String country = locale.getCountry();
        if (language.equals("zh") && country.equals("CN")) {
            return true;
        } else {
            return false;
        }
    }
}
