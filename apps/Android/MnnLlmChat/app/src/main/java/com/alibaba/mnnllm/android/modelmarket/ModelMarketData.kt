package com.alibaba.mnnllm.android.modelmarket

import com.google.gson.annotations.SerializedName

data class ModelMarketData(
    val tagMappings: Map<String, TagInfo>,
    val quickFilterTags: List<String>,
    val vendorOrder: List<String>? = emptyList(),
    val models: List<ModelMarketItem>,
    @SerializedName("tts_models") val ttsModels: List<ModelMarketItem>? = emptyList(),
    @SerializedName("asr_models") val asrModels: List<ModelMarketItem>? = emptyList()
)

data class TagInfo(
    val ch: String,
    val key: String
) 