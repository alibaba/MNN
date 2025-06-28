// Created by ruoyi.sjd on 2025/5/6.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.utils
import android.util.Log;
import com.taobao.meta.avatar.MHConfig

object LogUtils {
    fun v(tag: String, msg: String) {
        if (MHConfig.DEBUG_LOG_VERBOSE) {
            Log.v(tag, msg)
        }
    }
}