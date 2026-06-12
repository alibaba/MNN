package com.alibaba.mnnllm.android.chat.model

data class AgentSkillItem(
    val id: Long = 0L,
    val name: String,
    val description: String,
    val triggerKeywords: String,
    val actionTemplate: String,
    val enabled: Boolean = true,
    val createdAt: Long = System.currentTimeMillis()
)

