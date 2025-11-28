// Created by ruoyi.sjd on 2025/1/13.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.modelist

class ModelListContract {
    interface View {
        fun onListAvailable()
        fun onLoading()
        fun onListLoadError(error: String?)
        fun onBuiltinModelsCopyProgress(current: Int, total: Int, message: String)
        val adapter: ModelListAdapter?
        fun runModel(destPath:String?, modelId: String?)
    }
}
