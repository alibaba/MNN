package com.alibaba.mnnllm.android.modelmarket

import com.alibaba.mnnllm.android.utils.DeviceUtils

data class Tag(
    val ch: String,   // 中文显示
    val key: String   // 英文键值，用于过滤和记忆
) {
    fun getDisplayText(): String {
        return try {
            if (DeviceUtils.isChinese) ch else key
        } catch (e: Exception) {
            // Fallback for testing environment where DeviceUtils might not work
            key
        }
    }
} 