package com.alibaba.mnnllm.android.chat

import com.alibaba.mnnllm.android.chat.chatlist.ChatViewHolders

object ChatHistoryPersistencePolicy {
    fun shouldSaveInterruptedAssistant(
        isGenerating: Boolean,
        isMockStreamSession: Boolean,
        isDiffusion: Boolean,
        itemType: Int,
        text: String?
    ): Boolean {
        return isGenerating &&
            !isMockStreamSession &&
            !isDiffusion &&
            itemType == ChatViewHolders.ASSISTANT &&
            !text.isNullOrEmpty()
    }
}
