// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.utils

import android.annotation.SuppressLint
import com.alibaba.mls.api.ModelItem
import com.alibaba.mnnllm.android.R
import java.util.Locale

object ModelUtils {
    @JvmStatic
    fun getDrawableId(modelName: String?): Int {
        if (modelName == null) {
            return 0
        }
        val modelLower = modelName.lowercase(Locale.getDefault())
        if (modelLower.contains("deepseek")) {
            return R.drawable.deepseek_icon
        } else if (modelLower.contains("qwen") || modelLower.contains("qwq")) {
            return R.drawable.qwen_icon
        } else if (modelLower.contains("llama") || modelLower.contains("mobilellm")) {
            return R.drawable.llama_icon
        } else if (modelLower.contains("smo")) {
            return R.drawable.smolm_icon
        } else if (modelLower.contains("phi")) {
            return R.drawable.phi_icon
        } else if (modelLower.contains("baichuan")) {
            return R.drawable.baichuan_icon
        } else if (modelLower.contains("yi")) {
            return R.drawable.yi_icon
        } else if (modelLower.contains("glm") || modelLower.contains("codegeex")) {
            return R.drawable.chatglm_icon
        } else if (modelLower.contains("reader")) {
            return R.drawable.jina_icon
        } else if (modelLower.contains("internlm")) {
            return R.drawable.internlm_icon
        } else if (modelLower.contains("gemma")) {
            return R.drawable.gemma_icon
        }
        return 0
    }

    @SuppressLint("DefaultLocale")
    fun generateBenchMarkString(metrics: HashMap<String, Any>): String {
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
    fun generateDiffusionBenchMarkString(metrics: HashMap<String, Any>): String {
        val totalDuration = metrics["total_timeus"] as Long * 1.0 / 1000000.0
        return String.format("Generate time: %.2f s", totalDuration)
    }

    private val hotList: MutableSet<String> = HashSet()
    private val goodList: MutableSet<String> = HashSet()
    private val blackList: MutableSet<String> = HashSet()

    /**
     * you can add ModelItem.fromLocalModel("Qwen-Omni-7B", "/data/local/tmp/omni_test/model")
     * to load local models
     */
    private val localModelList = mutableListOf<ModelItem>()


    init {
        blackList.add("taobao-mnn/bge-large-zh-MNN") //embedding
        blackList.add("taobao-mnn/gte_sentence-embedding_multilingual-base-MNN") //embedding
        blackList.add("taobao-mnn/QwQ-32B-Preview-MNN") //too big
        blackList.add("taobao-mnn/codegeex2-6b-MNN") //not for chat
        blackList.add("taobao-mnn/chatglm-6b-MNN") //deprecated
        blackList.add("taobao-mnn/chatglm2-6b-MNN")
        blackList.add("taobao-mnn/stable-diffusion-v1-5-mnn-general") //in android, we use opencl version
    }

    init {
        hotList.add("taobao-mnn/DeepSeek-R1-7B-Qwen-MNN")
    }


    init {
        goodList.add("taobao-mnn/DeepSeek-R1-1.5B-Qwen-MNN")
        goodList.add("taobao-mnn/Qwen2.5-0.5B-Instruct-MNN")
        goodList.add("taobao-mnn/Qwen2.5-1.5B-Instruct-MNN")
        goodList.add("taobao-mnn/Qwen2.5-7B-Instruct-MNN")
        goodList.add("taobao-mnn/Qwen2.5-3B-Instruct-MNN")
        goodList.add("taobao-mnn/gemma-2-2b-it-MNN")
    }

    private fun isBlackListPattern(modelName: String): Boolean {
        return modelName.contains("qwen1.5")
                || modelName.contains("qwen-1")
                || isDiffusionModel(modelName) && (modelName.contains("metal") || modelName.contains(
            "gpu"
        ))
    }


    private fun isQwen3(modelName: String):Boolean {
        return modelName.lowercase(Locale.getDefault()).contains("qwen3")
    }

    fun processList(hfModelItems: List<ModelItem>): List<ModelItem> {

        val goodItems: MutableList<ModelItem> = ArrayList()
        val recommendedItems: MutableList<ModelItem> = ArrayList()
        val chatItems: MutableList<ModelItem> = ArrayList()
        val otherItems: MutableList<ModelItem> = ArrayList()
        for (item in hfModelItems) {
            val modelIdLowerCase = item.modelId!!.lowercase(Locale.getDefault())
            if (blackList.contains(item.modelId) || isBlackListPattern(modelIdLowerCase)) {
                continue
            }
            if (isQwen3(modelIdLowerCase)) {
                recommendedItems.add(item)
            } else if (goodList.contains(item.modelId)) {
                goodItems.add(item)
            } else if (modelIdLowerCase.contains("chat")) { //optimized for chat, should at top
                chatItems.add(item)
            } else {
                otherItems.add(item)
            }
        }
        val result: MutableList<ModelItem> = mutableListOf()
        result.addAll(localModelList)
        result.addAll(recommendedItems)
        result.addAll(goodItems)
        result.addAll(chatItems)
        result.addAll(otherItems)
        return result
    }

    fun isAudioModel(modelName: String): Boolean {
        return modelName.lowercase(Locale.getDefault()).contains("audio")
    }

    fun isMultiModalModel(modelName: String): Boolean {
        return isAudioModel(modelName) || isVisualModel(modelName) || isDiffusionModel(modelName)
    }

    fun isDiffusionModel(modelName: String): Boolean {
        return modelName.lowercase(Locale.getDefault()).contains("stable-diffusion")
    }

    @JvmStatic
    fun getModelName(modelId: String?): String? {
        if (modelId != null && modelId.contains("/")) {
            return modelId.substring(modelId.lastIndexOf("/") + 1)
        }
        return modelId
    }

    @JvmStatic
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

    fun isR1Model(modelName: String): Boolean {
        return modelName.lowercase(Locale.getDefault()).contains("deepseek-r1")
    }

    fun safeModelId(modelId: String): String {
        return modelId.replace("/".toRegex(), "_")
    }

    fun isOmni(modelName: String): Boolean {
        return modelName.lowercase(Locale.getDefault()).contains("omni")
    }

    fun isSupportThinkingSwitch(modelName: String): Boolean {
        return isQwen3(modelName)
    }
}
