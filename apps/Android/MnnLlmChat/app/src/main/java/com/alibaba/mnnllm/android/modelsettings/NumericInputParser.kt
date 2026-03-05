package com.alibaba.mnnllm.android.modelsettings

internal fun parseIntInput(text: CharSequence?): Int? {
    val value = text?.toString()?.trim().orEmpty()
    if (value.isEmpty()) {
        return null
    }
    return value.toIntOrNull()
}
