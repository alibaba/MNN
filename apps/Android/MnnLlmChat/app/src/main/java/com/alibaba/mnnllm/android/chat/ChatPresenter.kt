// Created by ruoyi.sjd on 2025/5/6.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.chat

import android.text.TextUtils
import android.util.Log
import androidx.lifecycle.lifecycleScope
import com.alibaba.mnnllm.android.ChatService
import com.alibaba.mnnllm.android.ChatSession
import com.alibaba.mnnllm.android.chat.ChatActivity.Companion.TAG
import com.alibaba.mnnllm.android.chat.model.ChatDataItem
import com.alibaba.mnnllm.android.chat.model.ChatDataManager
import com.alibaba.mnnllm.android.utils.ModelUtils
import kotlinx.coroutines.launch
import java.util.concurrent.Executors
import java.util.concurrent.ScheduledExecutorService

/**
 * ChatPresenter
 */
class ChatPresenter(
    private val chatActivity: ChatActivity,
    private val modelName: String,
    private val modelId: String
) {
    private var sessionId: String? = null
    private var sessionName:String? = null
    private var chatDataManager: ChatDataManager? = null
    private lateinit var chatSession: ChatSession
    private var chatExecutor: ScheduledExecutorService? = null
    init {
        chatDataManager = ChatDataManager.getInstance(chatActivity)
        chatExecutor = Executors.newScheduledThreadPool(1)
    }

    fun createSession():ChatSession {
        val intent = chatActivity.intent
        val chatService = ChatService.provide()
        sessionId = chatActivity.intent.getStringExtra("chatSessionId")
        val chatDataItemList: List<ChatDataItem>?
        if (!TextUtils.isEmpty(sessionId)) {
            chatDataItemList = chatDataManager!!.getChatDataBySession(sessionId!!)
            if (chatDataItemList.isNotEmpty()) {
                sessionName = chatDataItemList[0].text
            }
        } else {
            chatDataItemList = null
        }
        if (ModelUtils.isDiffusionModel(modelName)) {
            val diffusionDir = intent.getStringExtra("diffusionDir")
            chatSession = chatService.createDiffusionSession(
                modelId, diffusionDir,
                sessionId, chatDataItemList
            )
        } else {
            val configFilePath = intent.getStringExtra("configFilePath")
            chatSession = chatService.createSession(
                modelId, configFilePath, true,
                sessionId, chatDataItemList,
                ModelUtils.isOmni(modelName)
            )
        }
        sessionId = chatSession.sessionId
        chatSession.setKeepHistory(
            !ModelUtils.isVisualModel(modelName) && !ModelUtils.isAudioModel(
                modelName
            )
        )
        return chatSession
    }

    fun load() {
        Log.d(TAG, "current SessionId: $sessionId")
        chatExecutor!!.submit {
            Log.d(TAG, "chatSession loading")
            chatActivity.lifecycleScope.launch {
                chatActivity.setIsLoading(true)
            }
            chatSession.load()

            chatActivity.lifecycleScope.launch {
                chatActivity.setIsLoading(false)
            }
            Log.d(TAG, "chatSession loaded")
        }
    }
}