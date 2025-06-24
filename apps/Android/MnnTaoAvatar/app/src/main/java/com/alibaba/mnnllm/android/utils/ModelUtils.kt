// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.utils

import android.annotation.SuppressLint
import com.alibaba.mls.api.ModelItem
import java.util.Locale

object ModelUtils {
    @SuppressLint("DefaultLocale")
    fun generateBenchMarkString(metrics: HashMap<String?, Any?>): String {
        if (metrics.containsKey("total_timeus")) {
            return generateDiffusionBenchMarkString(metrics)
        }
        val promptLen = metrics["prompt_len"] as Long
        val decodeLen = metrics["decode_len"] as Long
        val prefillTimeUs = metrics["prefill_time"] as Long
        val decodeTimeUs = metrics["decode_time"] as Long
        // Calculate speeds in tokens per second
        val promptSpeed =
            if ((prefillTimeUs > 0)) (promptLen / (prefillTimeUs / 1000000.0)) else 0.0
        val decodeSpeed = if ((decodeTimeUs > 0)) (decodeLen / (decodeTimeUs / 1000000.0)) else 0.0
        return String.format(
            "Prefill: %d tokens, %.2f tokens/s\nDecode: %d tokens, %.2f tokens/s",
            promptLen, promptSpeed, decodeLen, decodeSpeed
        )
    }

    @SuppressLint("DefaultLocale")
    fun generateDiffusionBenchMarkString(metrics: HashMap<String?, Any?>): String {
        val totalDuration = metrics["total_timeus"] as Long * 1.0 / 1000000.0
        return String.format("Generate time: %.2f s", totalDuration)
    }

    private val blackList: MutableSet<String> = HashSet()

    init {
        blackList.add("taobao-mnn/bge-large-zh-MNN") //embedding
        blackList.add("taobao-mnn/gte_sentence-embedding_multilingual-base-MNN") //embedding
        blackList.add("taobao-mnn/QwQ-32B-Preview-MNN") //too big
        blackList.add("taobao-mnn/codegeex2-6b-MNN") //not for chat
        blackList.add("taobao-mnn/chatglm-6b-MNN") //deprecated
        blackList.add("taobao-mnn/chatglm2-6b-MNN")
        blackList.add("taobao-mnn/stable-diffusion-v1-5-mnn-general") //in android, we use opencl version
    }

    private val hotList: MutableSet<String> = HashSet()

    init {
        hotList.add("taobao-mnn/DeepSeek-R1-7B-Qwen-MNN")
    }

    private val goodList: MutableSet<String> = HashSet()

    init {
        goodList.add("taobao-mnn/DeepSeek-R1-1.5B-Qwen-MNN")
        goodList.add("taobao-mnn/Qwen2.5-0.5B-Instruct-MNN")
        goodList.add("taobao-mnn/Qwen2.5-1.5B-Instruct-MNN")
        goodList.add("taobao-mnn/Qwen2.5-7B-Instruct-MNN")
        goodList.add("taobao-mnn/Qwen2.5-3B-Instruct-MNN")
        goodList.add("taobao-mnn/gemma-2-2b-it-MNN")
    }

    private fun isBlackListPattern(modelName: String): Boolean {
        return modelName.contains("qwen1.5") || modelName.contains("qwen-1")
    }

    fun isAudioModel(modelName: String): Boolean {
        return modelName.lowercase(Locale.getDefault()).contains("audio")
    }

    fun isDiffusionModel(modelName: String): Boolean {
        return modelName.lowercase(Locale.getDefault()).contains("stable-diffusion")
    }

    fun getModelName(modelId: String?): String? {
        if (modelId != null && modelId.contains("/")) {
            return modelId.substring(modelId.lastIndexOf("/") + 1)
        }
        return modelId
    }

    fun generateSimpleTags(modelName: String, modelItem: ModelItem): ArrayList<String> {
        val splits = modelName.split("-".toRegex()).dropLastWhile { it.isEmpty() }.toTypedArray()
        val tags = ArrayList<String>()
        val isDiffusion = isDiffusionModel(modelName)
        if (splits.size > 1 && !isDiffusion) {
            val brand = splits[0]
            tags.add(brand.lowercase(Locale.getDefault()))
        }
        for (i in 1 until splits.size) {
            val tag = splits[i]
            if (tag.lowercase(Locale.getDefault()).matches("^[\\\\.0-9]+[mb]$".toRegex())) {
                tags.add(tag.lowercase(Locale.getDefault()))
            }
        }
        if (isDiffusion) {
            tags.add("diffusion")
        } else {
            tags.add("text")
            if (isAudioModel(modelName)) {
                tags.add("audio")
            } else if (isVisualModel(modelName)) {
                tags.add("visual")
            }
        }
        if (modelItem.isLocal) {
            tags.add("local")
        }
        return tags
    }

    fun isVisualModel(modelName: String): Boolean {
        return modelName.lowercase(Locale.getDefault()).contains("vl")
    }

    @JvmStatic
    fun isR1Model(modelName: String): Boolean {
        return modelName.lowercase(Locale.getDefault()).contains("deepseek-r1")
    }

    fun safeModelId(modelId: String): String {
        return modelId.replace("/".toRegex(), "_")
    }
}
