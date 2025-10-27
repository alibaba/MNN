// Created by ruoyi.sjd on 2025/2/10.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.utils

import android.os.Build
import com.alibaba.mls.api.ApplicationProvider

object DeviceUtils {
    val deviceInfo: String
        get() = ("""DeviceInfo: ${Build.MANUFACTURER} ${Build.MODEL} ${Build.VERSION.RELEASE}
SdkInt:${Build.VERSION.SDK_INT}""")

    @JvmStatic
    val isChinese: Boolean
        get() {
            val config =
                ApplicationProvider.get().resources.configuration
            val locale = config.locales[0]
            val language = locale.language
            val country = locale.country
            return if (language == "zh" && country == "CN") {
                true
            } else {
                false
            }
        }
}
