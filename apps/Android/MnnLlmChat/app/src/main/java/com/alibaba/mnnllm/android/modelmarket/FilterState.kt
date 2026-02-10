package com.alibaba.mnnllm.android.modelmarket

data class FilterState(
    val tagKeys: List<String> = emptyList(),
    val vendors: List<String> = emptyList(),
    val size: String? = null,
    val modality: String? = null,
    val downloadState: Int? = null,
    val source: String? = null,
    val searchQuery: String = ""
)