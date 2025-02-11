// Created by ruoyi.sjd on 2025/2/10.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.utils;
import android.os.Build;
public class DeviceUtils {

    public static String getDeviceInfo() {
        return "DeviceInfo: " + android.os.Build.MANUFACTURER + " "
                + android.os.Build.MODEL + " " + android.os.Build.VERSION.RELEASE +
                "\nSdkInt:" + Build.VERSION.SDK_INT;
    }
}
