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
     * Check if the given context's configuration indicates Chinese locale (zh_CN).
     * Use this for ViewHolder/UI bind to ensure correct locale at bind time.
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
        return locale.language == "zh" && locale.country == "CN"
    }
}
