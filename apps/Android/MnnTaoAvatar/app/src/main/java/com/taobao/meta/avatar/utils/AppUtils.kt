// Created by ruoyi.sjd on 2025/3/27.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.taobao.meta.avatar.utils;

import android.content.Context
import com.alibaba.mls.api.ApplicationProvider

public object AppUtils {

    fun getAppVersionName(context: Context): String {
        val packageManager = context.packageManager
        val packageInfo = packageManager.getPackageInfo(context.packageName, 0)
        return packageInfo.versionName!!
    }

    fun isChinese(): Boolean {
        val config = ApplicationProvider.get().resources.configuration
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
