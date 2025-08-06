package com.alibaba.mnnllm.android.modelmarket

import com.alibaba.mls.api.download.DownloadInfo
import com.google.gson.annotations.SerializedName

data class ModelMarketItem(
    val modelName: String,
    val vendor: String,
    @SerializedName("size_gb") val sizeB: Double, // Model parameters in billions, renamed from sizeGb for clarity
    val tags: List<String>, // Raw tags from JSON, will be converted to Tag objects
    val categories: List<String>,
    val sources: Map<String, String>,
    val description: String? = null,
    @SerializedName("file_size") val fileSize: Long = 0L, // File size in bytes from model_market.json
    var currentSource: String = "", // e.g. "modelscope", "huggingface"
    var currentRepoPath: String = "", // e.g. "MNN/Qwen-1.8B-Chat-Int4"
    var modelId: String = "" // e.g. "ModelScope/MNN/Qwen-1.8B-Chat-Int4"
) {
    // Convert string tags to Tag objects with i18n support
    val structuredTags: List<Tag>
        get() = tags.map { TagMapper.getTag(it) }
}

data class ModelMarketItemWrapper(
    val modelMarketItem: ModelMarketItem,
    var downloadInfo: DownloadInfo
) {
    fun copy(): ModelMarketItemWrapper {
        return ModelMarketItemWrapper(modelMarketItem, downloadInfo)
    }
} 