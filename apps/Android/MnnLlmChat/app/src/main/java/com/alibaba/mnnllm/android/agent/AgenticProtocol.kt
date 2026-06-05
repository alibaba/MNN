// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.agent

import com.google.gson.annotations.SerializedName

data class AgenticResponse(
    val reply: String? = null,
    @SerializedName("memory_updates") val memoryUpdates: List<AgentMemoryUpdate>? = null,
    @SerializedName("skill_updates") val skillUpdates: List<AgentSkillUpdate>? = null,
    @SerializedName("system_calls") val systemCalls: List<AgentSystemCall>? = null
) {
    fun hasToolCalls(): Boolean = !systemCalls.isNullOrEmpty()
    fun visibleReply(): String = reply.orEmpty()
}

data class AgentSystemCall(
    val type: String? = null,
    val query: String? = null,
    val url: String? = null,
    val code: String? = null,
    val input: String? = null,
    @SerializedName("timeout_ms") val timeoutMs: Long? = null,
    @SerializedName("input_files") val inputFiles: List<String>? = null,
    @SerializedName("expected_outputs") val expectedOutputs: List<String>? = null,
    @SerializedName("output_files") val outputFiles: List<String>? = null,
    @SerializedName("generated_files") val generatedFiles: List<String>? = null,
    val files: List<String>? = null
)

data class AgentMemoryUpdate(
    val category: String? = null,
    val content: String? = null
)

data class AgentSkillUpdate(
    val name: String? = null,
    val description: String? = null,
    @SerializedName("trigger_keywords") val triggerKeywords: List<String>? = null,
    @SerializedName("action_template") val actionTemplate: String? = null
)

data class AgentToolObservation(
    val type: String,
    val status: String,
    val title: String? = null,
    @SerializedName("final_url") val finalUrl: String? = null,
    val text: String? = null,
    val stdout: String? = null,
    val stderr: String? = null,
    @SerializedName("output_files") val outputFiles: List<String>? = null,
    val error: String? = null
)
