package com.alibaba.mnnllm.android.modelmarket

data class FilterState(
    val tagKeys: List<String> = emptyList(), // Changed to use tag keys (English) for filtering
    val vendors: List<String> = emptyList(),
    val size: String? = null,
    val modality: String? = null,
    val downloadState: String? = null,
    val source: String? = null,
    val searchQuery: String = ""
)