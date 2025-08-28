// Created by ruoyi.sjd on 2025/5/22.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.model

import com.alibaba.mnnllm.android.model.ModelUtils.isAudioModel
import com.alibaba.mnnllm.android.model.ModelUtils.isVisualModel

object Modality {
    const val Text = "Text"
    const val Visual = "Visual"
    const val Audio = "Audio"
    const val Omni = "Omni"
    const val Diffusion = "Diffusion"

    fun checkModality(modelId: String, modality:String): Boolean {
        if (modality == Text) {
            return true
        } else if (modality == Visual) {
            return isVisualModel(modelId)
        } else if (modality == Audio) {
            return isAudioModel(modelId)
        } else if (modality == Omni) {
            return ModelUtils.isOmni(modelId)
        } else if (modality == Diffusion) {
            return ModelUtils.isDiffusionModel(modelId)
        }
        return false
    }

    val modalitySelectorList = listOf(Visual, Audio, Omni, Diffusion)
}