// Created by ruoyi.sjd on 2025/3/12.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.taobao.meta.avatar.record

import android.Manifest


object RecordPermission {
    const val REQUEST_RECORD_AUDIO_PERMISSION = 200
    val permissions = arrayOf(Manifest.permission.RECORD_AUDIO)
}