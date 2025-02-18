// Created by ruoyi.sjd on 2024/12/18.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mls.api;

import android.app.Application;

public class ApplicationUtils {
    static Application application;
    public static void set(Application application ) {
        ApplicationUtils.application = application;
    }

    public static Application get() {
        return application;
    }
}
