// Created by ruoyi.sjd on 2025/5/6.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.chat

import android.text.TextUtils
import android.util.Log
import androidx.lifecycle.lifecycleScope
import com.alibaba.mnnllm.android.ChatService
import com.alibaba.mnnllm.android.ChatSession
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.chat.ChatActivity.Companion.TAG
import com.alibaba.mnnllm.android.chat.GenerateResultProcessor.R1GenerateResultProcessor
import com.alibaba.mnnllm.android.chat.model.ChatDataItem
import com.alibaba.mnnllm.android.chat.model.ChatDataManager
import com.alibaba.mnnllm.android.utils.FileUtils
import com.alibaba.mnnllm.android.utils.ModelUtils
import kotlinx.coroutines.launch
import java.util.Random
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
    private var stopGenerating = false
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
                chatActivity.onLoadingChanged(true)
            }
            chatSession.load()

            chatActivity.lifecycleScope.launch {
                chatActivity.onLoadingChanged(false)
            }
            Log.d(TAG, "chatSession loaded")
        }
    }

    fun reset(onResetSuccess: (newSessionId: String) -> Unit) {
        chatExecutor!!.submit {
            chatDataManager!!.deleteAllChatData(sessionId!!)
            sessionId = chatSession.reset()
            chatActivity.lifecycleScope.launch {
                onResetSuccess(sessionId!!)
            }
        }
    }

    private fun submitDiffusionRequest(prompt:String): HashMap<String, Any> {
        val diffusionDestPath = FileUtils.generateDestDiffusionFilePath(
            chatActivity,
            sessionId!!
        )
        return chatSession.generateDiffusion(
            prompt, diffusionDestPath, 20,
            Random(System.currentTimeMillis()).nextInt(), object : ChatSession.GenerateProgressListener {
                override fun onProgress(progress: String?): Boolean {
                    chatActivity.lifecycleScope.launch {
                        chatActivity.onDiffusionGenerateProgress(progress, diffusionDestPath)
                    }
                    return false
                }
            }
        )
    }

    private fun submitLlmRequest(prompt:String): HashMap<String, Any> {
        val generateResultProcessor: GenerateResultProcessor =
            R1GenerateResultProcessor(
                chatActivity.getString(R.string.r1_thinking_message),
                chatActivity.getString(R.string.r1_think_complete_template))
        generateResultProcessor.generateBegin()
        return chatSession.generate(prompt, object: ChatSession.GenerateProgressListener {
            override fun onProgress(progress: String?): Boolean {
                generateResultProcessor.process(progress)
                chatActivity.lifecycleScope.launch {
                    chatActivity.onLlmGenerateProgress(progress, generateResultProcessor)
                }
                if (stopGenerating) {
                    Log.d(TAG, "stopGenerating requested")
                }
                return stopGenerating
            }
        })
    }

    private fun submitRequest(input: String, userData: ChatDataItem) {
        stopGenerating = false
        val benchMarkResult = if (ModelUtils.isDiffusionModel(this.modelName)) {
            submitDiffusionRequest(input)
        } else {
            submitLlmRequest(input)
        }
        chatActivity.lifecycleScope.launch {
            chatActivity.onGenerateFinished(benchMarkResult)
        }
        chatDataManager!!.addChatData(sessionId, userData)
    }

    private fun updateSession(sessionId: String, modelId: String?, sessionName: String) {
        chatDataManager!!.addOrUpdateSession(sessionId!!, modelId)
        chatDataManager!!.updateSessionName(this.sessionId!!, this.sessionName)
    }

    fun onUserMessage(userData: ChatDataItem) {
        val prompt =  PromptUtils.generateUserPrompt(userData)
        if (this.sessionName.isNullOrEmpty()) {
            this.sessionName = SessionUtils.generateSessionName(userData)
            updateSession(sessionId!!, modelId, sessionName!!)
        }
        chatDataManager!!.addChatData(sessionId, userData)
        chatActivity.onGenerateStart()
        chatExecutor!!.execute { submitRequest(prompt, userData) }
    }

    fun stopGenerate() {
        stopGenerating = true
    }

    fun destroy() {
        stopGenerate()
        chatExecutor!!.submit {
            chatSession.reset()
            chatSession.release()
            chatExecutor!!.shutdownNow()
        }
    }
}