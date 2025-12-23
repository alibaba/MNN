package com.alibaba.mnn.tts.demo

data class ModelConfig(
    val speakers: List<String> = emptyList(),
    val languages: List<String> = emptyList()
)
