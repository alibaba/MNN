package com.alibaba.mnnllm.android.chat.chatlist

import com.alibaba.mnnllm.android.chat.model.ChatDataItem

object AssistantTextRenderPolicy {
    fun usePlainText(data: ChatDataItem): Boolean {
        return data.forceShowLoadingWithText
    }
}
