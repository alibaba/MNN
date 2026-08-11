package com.alibaba.mnnllm.android.chat

object DiffusionProgressHintPolicy {
    fun isMeaningfulProgress(progress: String?): Boolean {
        return (progress?.toIntOrNull() ?: 0) > 0
    }
}
