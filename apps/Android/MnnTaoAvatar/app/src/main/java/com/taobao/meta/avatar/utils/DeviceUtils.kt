// Created by ruoyi.sjd on 2025/3/27.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.taobao.meta.avatar.utils

import android.content.res.Configuration
import android.os.Build
import com.alibaba.mls.api.ApplicationProvider
import java.util.Locale

object DeviceUtils {
    val deviceInfo: String
        get() = ("DeviceInfo: " + Build.MANUFACTURER + " "
                + Build.MODEL + " " + Build.VERSION.RELEASE +
                "\nSdkInt:" + Build.VERSION.SDK_INT)

    val isChinese: Boolean
        get() {
            val config: Configuration = ApplicationProvider.get().resources.configuration
            val locale: Locale = config.locales.get(0)
            val language: String = locale.language
            val country: String = locale.country
            if (language == "zh" && country == "CN") {
                return true
            } else {
                return false
            }
        }
}