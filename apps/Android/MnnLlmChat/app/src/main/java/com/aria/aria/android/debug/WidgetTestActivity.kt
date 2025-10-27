// Created by ruoyi.sjd on 2025/7/1.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.debug

import android.os.Bundle
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import com.alibaba.mnnllm.android.R
import com.jaredrummler.android.device.DeviceName

class WidgetTestActivity: AppCompatActivity()  {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.layout_widgets_test)
    }

    fun getDeviceName(view: View) {
        DeviceName.with(this).request { info, error ->
            val deviceName = info?.marketName ?: info?.name ?: android.os.Build.MODEL
        }
    }
}