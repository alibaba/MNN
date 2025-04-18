// Created by ruoyi.sjd on 2024/12/18.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android

import android.app.Application
import com.alibaba.mls.api.ApplicationProvider

class MnnLlmApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        ApplicationProvider.set(this)
    }
}
