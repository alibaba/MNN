package com.alibaba.mnnllm.android.llm

import com.alibaba.mnnllm.android.chat.chatlist.ChatViewHolders
import com.alibaba.mnnllm.android.chat.model.ChatDataItem

/**
 * Native rebuilds the conversation from a flat list, reading even positions as user turns
 * and odd positions as replies. Stored history is not always that regular: a stopped turn
 * can leave a user row without a reply, a header row may sit in front. This turns it into
 * strict user/assistant pairs so roles never shift.
 */
object NativeHistory {
    fun alternate(items: List<ChatDataItem>): List<String> {
        val result = ArrayList<String>(items.size)
        var expectUser = true
        for (item in items) {
            val text = item.text ?: continue
            when (item.type) {
                ChatViewHolders.USER -> {
                    if (!expectUser) {
                        result.add("")  // a user turn without a reply
                    }
                    result.add(text)
                    expectUser = false
                }
                ChatViewHolders.ASSISTANT -> {
                    if (expectUser) {
                        continue  // a reply without its request carries no usable context
                    }
                    result.add(text)
                    expectUser = true
                }
            }
        }
        return result
    }
}
