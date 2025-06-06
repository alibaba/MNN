// Created by ruoyi.sjd on 2024/12/18.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android

import android.app.Application
import com.alibaba.mls.api.ApplicationProvider
import com.alibaba.mnnllm.android.utils.CrashUtil
import timber.log.Timber
import android.content.Context



class MnnLlmApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        ApplicationProvider.set(this)
        CrashUtil.init(this)
        instance = this

        //Application 初始化时种下日志：
        Timber.plant(Timber.DebugTree())

    }
    companion object {
        private lateinit var instance: MnnLlmApplication

        fun getAppContext(): Context {
            return instance.applicationContext
        }
    }
}
