// Created by ruoyi.sjd on 2025/2/28.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.utils

import android.content.Context
import android.content.Intent

object RouterUtils {
    fun startActivity(context: Context, cls: Class<*>) {
        context.startActivity(Intent(context, cls))
    }
}