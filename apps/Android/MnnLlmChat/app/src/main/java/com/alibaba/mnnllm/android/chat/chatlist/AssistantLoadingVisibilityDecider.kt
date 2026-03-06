package com.alibaba.mnnllm.android.chat.chatlist

import com.alibaba.mnnllm.android.chat.model.ChatDataItem

object AssistantLoadingVisibilityDecider {
    fun shouldShow(data: ChatDataItem): Boolean {
        if (data.hasOmniAudio) {
            return data.loading
        }
        if (data.loading && data.forceShowLoadingWithText) {
            return true
        }
        return data.displayText.isNullOrEmpty() && data.thinkingText.isNullOrEmpty()
    }
}
