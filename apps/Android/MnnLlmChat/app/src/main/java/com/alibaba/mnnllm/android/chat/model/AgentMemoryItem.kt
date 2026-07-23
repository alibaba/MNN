package com.alibaba.mnnllm.android.chat.model

data class AgentMemoryItem(
    val id: Long = 0L,
    val category: String,
    val content: String,
    val source: String = "agent",
    val updatedAt: Long = System.currentTimeMillis()
)

