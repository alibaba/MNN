// Created by ruoyi.sjd on 2025/5/6.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.chat

import android.text.TextUtils
import android.util.Log
import androidx.lifecycle.lifecycleScope
import com.alibaba.mnnllm.android.llm.ChatService
import com.alibaba.mnnllm.android.llm.ChatSession
import com.alibaba.mnnllm.android.chat.ChatActivity.Companion.TAG
import com.alibaba.mnnllm.android.chat.model.ChatDataItem
import com.alibaba.mnnllm.android.chat.model.ChatDataManager
import com.alibaba.mnnllm.android.llm.GenerateProgressListener
import com.alibaba.mnnllm.android.utils.FileUtils
import com.alibaba.mnnllm.android.model.ModelUtils
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.async
import kotlinx.coroutines.cancel
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
    var stopGenerating = false
    private var sessionId: String? = null
    private var sessionName:String? = null
    private var chatDataManager: ChatDataManager? = null
    private lateinit var chatSession: ChatSession
    private val presenterScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private var generateListener:GenerateListener? = null

    init {
        chatDataManager = ChatDataManager.getInstance(chatActivity)
    }

    fun createSession(): ChatSession {
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
            chatSession = chatService.createLlmSession(
                modelId, configFilePath,
                sessionId, chatDataItemList,
                ModelUtils.isOmni(modelName)
            )
        }
        sessionId = chatSession.sessionId
        chatSession.setKeepHistory(
            true
        )
        return chatSession
    }

    fun load() {
        Log.d(TAG, "current SessionId: $sessionId")
        presenterScope.launch {
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
        presenterScope.launch {
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
        return chatSession.generate(
            prompt,
            mapOf(
                "output" to diffusionDestPath,
                "iterNum" to 20,
                "randomSeed" to Random(System.currentTimeMillis()).nextInt()
            )
            , object : GenerateProgressListener {
                override fun onProgress(progress: String?): Boolean {
                    chatActivity.lifecycleScope.launch {
                        this@ChatPresenter.generateListener?.onDiffusionGenerateProgress(progress, diffusionDestPath)
                    }
                    return false
                }
            }
        )
    }

    private fun submitLlmRequest(prompt:String): HashMap<String, Any> {
        val generateResultProcessor =
            GenerateResultProcessor()
        generateResultProcessor.generateBegin()
        val result = chatSession.generate(prompt, mapOf(), object: GenerateProgressListener {
            override fun onProgress(progress: String?): Boolean {
                generateResultProcessor.process(progress)
                chatActivity.lifecycleScope.launch {
                    this@ChatPresenter.generateListener?.onLlmGenerateProgress(progress, generateResultProcessor)
                }
                if (stopGenerating) {
                    Log.d(TAG, "stopGenerating requested")
                }
                return stopGenerating
            }
        })
        result["response"] = generateResultProcessor.getRawResult()
        return result
    }

    private fun submitRequest(input: String, userData: ChatDataItem): HashMap<String, Any> {
        stopGenerating = false
        val benchMarkResult = if (ModelUtils.isDiffusionModel(this.modelName)) {
            submitDiffusionRequest(input)
        } else {
            submitLlmRequest(input)
        }
        chatActivity.lifecycleScope.launch {
            this@ChatPresenter.generateListener?.onGenerateFinished(benchMarkResult)
        }
        return benchMarkResult
    }

    private fun updateSession(sessionId: String, modelId: String?, sessionName: String) {
        chatDataManager!!.addOrUpdateSession(sessionId, modelId)
        chatDataManager!!.updateSessionName(this.sessionId!!, this.sessionName)
    }


    suspend fun requestGenerate(userData: ChatDataItem, generateListener: GenerateListener): HashMap<String, Any> {
        this.generateListener = generateListener
        val prompt =  PromptUtils.generateUserPrompt(userData)
        if (this.sessionName.isNullOrEmpty()) {
            this.sessionName = SessionUtils.generateSessionName(userData)
            updateSession(sessionId!!, modelId, sessionName!!)
        }
        chatDataManager!!.addChatData(sessionId, userData)
        this.generateListener?.onGenerateStart()
        val result =  presenterScope.async {
            return@async submitRequest(prompt, userData)
        }.await()
        return result
    }

    fun stopGenerate() {
        stopGenerating = true
    }

    fun destroy() {
        stopGenerate()
        presenterScope.cancel("ChatPresenter destroy")
        presenterScope.launch {
            chatSession.reset()
            chatSession.release()
        }
        CoroutineScope(Dispatchers.IO + SupervisorJob()).launch {
            try {
                if (::chatSession.isInitialized) {
                    Log.d(TAG, "Final cleanup: Resetting and releasing chat session.")
                    chatSession.reset()
                    chatSession.release()
                    Log.d(TAG, "Chat session reset and released during destroy.")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error during final chat session cleanup", e)
            }
        }
    }

    fun saveResponseToDatabase(recentItem: ChatDataItem) {
        this.chatDataManager?.addChatData(sessionId, recentItem)
    }

    fun setEnableAudioOutput(enable: Boolean) {
        this.chatSession.setEnableAudioOutput(enable)
    }

    interface GenerateListener {
        fun onDiffusionGenerateProgress(progress: String?, diffusionDestPath: String?)
        fun onGenerateStart()
        fun onGenerateFinished(benchMarkResult: HashMap<String, Any>)
        fun onLlmGenerateProgress(progress: String?, generateResultProcessor: GenerateResultProcessor)
    }
}