package com.alibaba.mnnllm.android.modelmarket

import com.google.gson.annotations.SerializedName

data class ModelMarketData(
    val tagTranslations: Map<String, String>,
    val quickFilterTags: List<String>,
    val vendorOrder: List<String>? = emptyList(),
    val models: List<ModelMarketItem>,
    @SerializedName("tts_models") val ttsModels: List<ModelMarketItem>? = emptyList(),
    @SerializedName("asr_models") val asrModels: List<ModelMarketItem>? = emptyList()
) 