package com.alibaba.mnnllm.android.chat

import java.util.Locale

object ContextUsageFormat {
    /** 950 -> "950", 1234 -> "1.2K", 4096 -> "4K", 32768 -> "32K". */
    fun tokens(count: Int): String {
        if (count < 1000) {
            return count.coerceAtLeast(0).toString()
        }
        val thousands = count / 1024.0
        return if (count % 1024 == 0 || thousands >= 100) {
            String.format(Locale.US, "%dK", Math.round(thousands))
        } else {
            String.format(Locale.US, "%.1fK", thousands)
        }
    }
}
