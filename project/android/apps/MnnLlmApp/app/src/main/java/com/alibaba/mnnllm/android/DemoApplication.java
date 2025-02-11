// Created by ruoyi.sjd on 2024/12/18.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android;

import android.app.Application;

import com.alibaba.mls.api.ApplicationUtils;

public class DemoApplication extends Application {
    @Override
    public void onCreate() {
        super.onCreate();
        ApplicationUtils.set(this);
    }
}
