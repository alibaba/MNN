// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android

import android.text.TextUtils
import com.alibaba.mnnllm.android.chat.ChatDataItem

class ChatService {
    private val transformerSessionMap: MutableMap<String, ChatSession> = HashMap()
    private val diffusionSessionMap: MutableMap<String, ChatSession> = HashMap()

    @Synchronized
    fun createSession(
        modelId: String?,
        modelDir: String?,
        useTmpPath: Boolean,
        sessionId: String,
        chatDataItemList: List<ChatDataItem>?
    ): ChatSession {
        var sessionId = sessionId
        if (TextUtils.isEmpty(sessionId)) {
            sessionId = System.currentTimeMillis().toString()
        }
        val session = ChatSession(modelId!!, sessionId, modelDir!!, useTmpPath, chatDataItemList)
        transformerSessionMap[sessionId] = session
        return session
    }

    @Synchronized
    fun createDiffusionSession(
        modelId: String?,
        modelDir: String?,
        sessionId: String,
        chatDataItemList: List<ChatDataItem>?
    ): ChatSession {
        var sessionId = sessionId
        if (TextUtils.isEmpty(sessionId)) {
            sessionId = System.currentTimeMillis().toString()
        }
        val session = ChatSession(modelId!!, sessionId, modelDir!!, false, chatDataItemList, true)
        diffusionSessionMap[sessionId] = session
        return session
    }

    @Synchronized
    fun getSession(sessionId: String): ChatSession? {
        return if (transformerSessionMap.containsKey(sessionId)) {
            transformerSessionMap[sessionId]
        } else {
            diffusionSessionMap[sessionId]
        }
    }

    @Synchronized
    fun removeSession(sessionId: String) {
        transformerSessionMap.remove(sessionId)
        diffusionSessionMap.remove(sessionId)
    }

    companion object {
        private var instance: ChatService? = null

        @JvmStatic
        @Synchronized
        fun provide(): ChatService {
            if (instance == null) {
                instance = ChatService()
            }
            return instance!!
        }
    }
}
