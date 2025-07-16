// Created by ruoyi.sjd on 2024/12/18.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android

import android.app.Application
import com.alibaba.mls.api.ApplicationProvider
import com.alibaba.mnnllm.android.utils.CrashUtil
import com.alibaba.mnnllm.android.utils.CurrentActivityTracker
import timber.log.Timber
import android.content.Context
import com.alibaba.mls.api.ModelTagsCache
import com.jaredrummler.android.device.DeviceName

class MnnLlmApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        ApplicationProvider.set(this)
        CrashUtil.init(this)
        instance = this
        DeviceName.init(this)

        // Initialize CurrentActivityTracker
        CurrentActivityTracker.initialize(this)

        Timber.plant(Timber.DebugTree())

        // Initialize model tags cache for proper tag loading
        ModelTagsCache.initializeCache(this)
    }

    companion object {
        private lateinit var instance: MnnLlmApplication

        fun getAppContext(): Context {
            return instance.applicationContext
        }
    }
}
