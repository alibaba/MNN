// Created by ruoyi.sjd on 2024/12/18.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android

import android.app.Application
import com.alibaba.mls.api.ApplicationProvider
import com.alibaba.mnnllm.android.utils.CrashUtil
import com.alibaba.mnnllm.android.utils.CurrentActivityTracker
import com.alibaba.mnnllm.android.utils.TimberConfig
import timber.log.Timber
import android.content.Context
import com.jaredrummler.android.device.DeviceName
import com.alibaba.mnnllm.android.modelist.ModelListManager

class MnnLlmApplication : Application() {
    
    override fun onCreate() {
        super.onCreate()
        ApplicationProvider.set(this)
        CrashUtil.init(this)
        instance = this
        DeviceName.init(this)

        // Initialize CurrentActivityTracker
        CurrentActivityTracker.initialize(this)

        // Initialize Timber logging based on configuration
        TimberConfig.initialize()
        
        // Set context for ModelListManager (enables auto-initialization)
        ModelListManager.setContext(getInstance())
    }

    companion object {
        private lateinit var instance: MnnLlmApplication

        fun getAppContext(): Context {
            return instance.applicationContext
        }
        
        /**
         * Get the application instance for accessing Timber configuration
         */
        fun getInstance(): MnnLlmApplication {
            return instance
        }
    }
}
