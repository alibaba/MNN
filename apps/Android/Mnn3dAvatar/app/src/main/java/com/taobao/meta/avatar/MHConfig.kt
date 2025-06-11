// Created by ruoyi.sjd on 2025/3/12.
// Copyright (c) 2024 Alibaba Group Holding Lim
// ited All rights reserved.
package com.taobao.meta.avatar

import android.bluetooth.BluetoothClass.Device
import com.taobao.meta.avatar.utils.DeviceUtils


object MHConfig {
    private var baseDir: String = ""
    var BASE_DIR
        get() = baseDir
        set(value) {
            baseDir = value
        }
    val NNR_MODEL_DIR
        get() = "${BASE_DIR}/TaoAvatar-NNR-MNN/"

    val TTS_MODEL_DIR
        get() = "${BASE_DIR}/bert-vits2-MNN/"

    val A2BS_MODEL_DIR
        get() = "${BASE_DIR}/UniTalker-MNN/"

    val LLM_MODEL_DIR
        get() = "${BASE_DIR}/Qwen2.5-1.5B-Instruct-MNN"

    val ASR_MODEL_DIR
        get() = if (DeviceUtils.isChinese) {
            "${BASE_DIR}/sherpa-mnn-streaming-zipformer-bilingual-zh-en-2023-02-20/"
        } else {
            "${BASE_DIR}/sherpa-mnn-streaming-zipformer-en-2023-02-21"
        }

    const val DEBUG_MODE = false

    const val DEBUG_SCREEN_SHOT = false

    const val DEBUG_LOG_VERBOSE = false

    object DebugConfig  {
        val DebugWriteBlendShape = false
    }
}