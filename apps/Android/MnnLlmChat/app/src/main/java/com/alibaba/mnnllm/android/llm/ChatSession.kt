// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.llm

interface ChatSession  {
    val debugInfo: String
    val sessionId: String?

    val supportOmni: Boolean
    fun load()

    fun generate(prompt: String, params: Map<String, Any>, progressListener: GenerateProgressListener): HashMap<String, Any>

    fun reset(): String

    fun release()
    fun setKeepHistory(keepHistory: Boolean)
    fun setEnableAudioOutput(enable: Boolean)
}
