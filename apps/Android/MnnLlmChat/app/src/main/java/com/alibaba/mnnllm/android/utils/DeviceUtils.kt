// Created by ruoyi.sjd on 2025/2/10.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.utils

import android.content.Context
import android.os.Build
import com.alibaba.mls.api.ApplicationProvider

object DeviceUtils {
    val deviceInfo: String
        get() = ("""DeviceInfo: ${Build.MANUFACTURER} ${Build.MODEL} ${Build.VERSION.RELEASE}
SdkInt:${Build.VERSION.SDK_INT}""")

    @JvmStatic
    val isChinese: Boolean
        get() = isChinese(ApplicationProvider.get())

    /**
     * Check whether the given context's configuration is a Chinese locale.
     *
     * Any `zh` region counts, not just `zh_CN`: zh_TW/zh_HK users read the Chinese strings too, and
     * callers use this to pick a Chinese default model source and Chinese update notes. Comparing
     * the country to "CN" alone sent those users to the English paths.
     */
    @JvmStatic
    fun isChinese(context: Context): Boolean {
        val config = context.resources.configuration
        val locale = if (Build.VERSION.SDK_INT >= 24) {
            if (config.locales.isEmpty()) null else config.locales.get(0)
        } else {
            @Suppress("DEPRECATION")
            config.locale
        } ?: return false
        return locale.language == "zh"
    }
}
