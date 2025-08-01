package com.alibaba.mnnllm.android.utils

interface Searchable {
    fun onSearchQuery(query: String)
    fun onSearchCleared()
} 