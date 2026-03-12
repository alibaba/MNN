package com.alibaba.mnnllm.android.modelmarket

import com.alibaba.mnnllm.android.utils.DeviceUtils

data class Tag(
    val ch: String,   //Chinese display
    val key: String   //English key for filtering and memory
) {
    fun getDisplayText(): String {
        return try {
            if (DeviceUtils.isChinese) ch else key
        } catch (e: Exception) {
            // Fallback for testing environment where DeviceUtils might not work
            key
        }
    }

    /** Use when context is available for locale-aware display (e.g. ViewHolder bind). */
    fun getDisplayText(useChinese: Boolean): String = if (useChinese) ch else key
} 