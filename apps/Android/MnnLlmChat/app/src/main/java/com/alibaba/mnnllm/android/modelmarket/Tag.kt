package com.alibaba.mnnllm.android.modelmarket

import android.content.Context
import com.alibaba.mls.api.ApplicationProvider
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.utils.DeviceUtils

data class Tag(
    val ch: String,   //Chinese display
    val key: String   //English key for filtering and memory
) {
    fun getDisplayText(): String {
        return try {
            getDisplayText(ApplicationProvider.get())
        } catch (e: Exception) {
            // Fallback for testing environment where DeviceUtils might not work
            key
        }
    }

    fun getDisplayText(context: Context): String {
        val resourceId = when (key) {
            "Chat" -> R.string.model_tag_chat
            "Think" -> R.string.model_tag_think
            "Vision" -> R.string.model_tag_vision
            "ImageGen" -> R.string.model_tag_image_gen
            "Audio" -> R.string.model_tag_audio
            "AudioGen" -> R.string.model_tag_audio_gen
            "Code" -> R.string.model_tag_code
            "Math" -> R.string.model_tag_math
            "Video" -> R.string.model_tag_video
            "Safety" -> R.string.model_tag_safety
            else -> null
        }
        return resourceId?.let { context.getString(it) } ?: getDisplayText(DeviceUtils.isChinese(context))
    }

    fun getDisplayText(useChinese: Boolean): String = if (useChinese) ch else key
} 