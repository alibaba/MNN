// Created by ruoyi.sjd on 2024/12/18.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mls.api

import android.app.Application

object ApplicationProvider {
    var application: Application? = null
    fun set(application: Application?) {
        ApplicationProvider.application = application
    }

    @JvmStatic
    fun get(): Application {
        return application!!
    }
}
