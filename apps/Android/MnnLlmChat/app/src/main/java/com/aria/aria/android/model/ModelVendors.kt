// Created by ruoyi.sjd on 2025/5/22.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.model

object ModelVendors {
    const val Qwen = "Qwen"
    const val DeepSeek = "DeepSeek"
    const val Llama = "Llama"
    const val Smo = "Smo"
    const val Phi = "Phi"
    const val Baichuan = "Baichuan"
    const val Yi = "Yi"
    const val Glm = "Glm"
    const val Jina = "Jina"
    const val Internlm = "Internlm"
    const val Gemma = "Gemma"
    const val Mimo = "Mimo"
    const val FastVlm = "FastVlm"
    const val OpenElm = "OpenElm"
    const val Others = "Others"

    val vendorList = listOf(
        Qwen,
        DeepSeek,
        Gemma,
        Smo,
        FastVlm,
        Phi,
        Mimo,
        Llama,
        Yi,
        Glm,
        Jina,
        OpenElm,
        Internlm,
        Baichuan,
        Others
    )
}