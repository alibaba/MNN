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
    @SerializedName("min_app_version") val minAppVersion: String? = null,
    val description: String? = null,
    @SerializedName("file_size") val fileSize: Long = 0L, // File size in bytes from model_market.json
    @SerializedName("extra_tags") val extraTags: List<String> = emptyList(), // Extra tags not shown to users
    var currentSource: String = "", // e.g. "modelscope", "huggingface"
    var currentRepoPath: String = "", // e.g. "MNN/Qwen-1.8B-Chat-Int4"
    var modelId: String = "" // e.g. "ModelScope/MNN/Qwen-1.8B-Chat-Int4"
) {
    // Convert string tags to Tag objects with i18n support
    val structuredTags: List<Tag>
        get() = TagMapper.getCapabilityTags(tags)

    val displayVendor: String
        get() = if (
            modelName.startsWith("MNN-Pocket", ignoreCase = true) &&
            (vendor.isBlank() || vendor.equals("Other", ignoreCase = true) ||
                vendor.equals("MNN-Pocket", ignoreCase = true))
        ) {
            "MNN"
        } else {
            vendor
        }
}

data class ModelMarketItemWrapper(
    val modelMarketItem: ModelMarketItem,
    var downloadInfo: DownloadInfo
) {
    fun copy(): ModelMarketItemWrapper {
        return ModelMarketItemWrapper(modelMarketItem, downloadInfo)
    }
} 
