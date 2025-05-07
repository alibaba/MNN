// Created by ruoyi.sjd on 2025/5/6.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.chat

import android.text.TextUtils
import android.util.Log
import com.alibaba.mnnllm.android.ChatService
import com.alibaba.mnnllm.android.ChatSession
import com.alibaba.mnnllm.android.audio.AudioPlayer
import com.alibaba.mnnllm.android.chat.ChatActivity.Companion.TAG
import com.alibaba.mnnllm.android.utils.ModelUtils
import kotlinx.coroutines.MainScope
import kotlinx.coroutines.launch

/**
 * ChatPresenter
 * TODO: move the logics to here
 */
class ChatPresenter(private val chatActivity: ChatActivity) {
    private var sessionId: String? = null
    private var sessionName:String? = null
    private fun setupSession() {
//        val chatService = ChatService.provide()
//        sessionId = chatActivity.intent.getStringExtra("chatSessionId")
//        val chatDataItemList: List<ChatDataItem>?
//        if (!TextUtils.isEmpty(sessionId)) {
//            chatDataItemList = chatDataManager!!.getChatDataBySession(sessionId!!)
//            if (chatDataItemList.isNotEmpty()) {
//                sessionName = chatDataItemList[0].text
//            }
//        } else {
//            chatDataItemList = null
//        }
//        if (ModelUtils.isDiffusionModel(modelName!!)) {
//            val diffusionDir = intent.getStringExtra("diffusionDir")
//            chatSession = chatService.createDiffusionSession(
//                modelId, diffusionDir,
//                sessionId, chatDataItemList
//            )
//        } else {
//            val configFilePath = intent.getStringExtra("configFilePath")
//            chatSession = chatService.createSession(
//                modelId, configFilePath, true,
//                sessionId, chatDataItemList,
//                ModelUtils.isOmni(modelName!!)
//            )
//        }
//        sessionId = chatSession.sessionId
//        chatSession.setKeepHistory(
//            !ModelUtils.isVisualModel(modelName!!) && !ModelUtils.isAudioModel(
//                modelName!!
//            )
//        )
//        Log.d(TAG, "current SessionId: $sessionId")
//        chatExecutor!!.submit {
//            Log.d(TAG, "chatSession loading")
//            setIsLoading(true)
//            chatSession.load()
//            if (chatSession.supportOmni) {
//                audioPlayer = AudioPlayer()
//                audioPlayer!!.start()
//                chatSession.setAudioDataListener(object : ChatSession.AudioDataListener {
//                    override fun onAudioData(data: FloatArray, isEnd: Boolean): Boolean {
//                        MainScope().launch {
//                            audioPlayer?.playChunk(data)
//                        }
//                        return true
//                    }
//                })
//            }
//            setIsLoading(false)
//            Log.d(TAG, "chatSession loaded")
//        }
    }
}