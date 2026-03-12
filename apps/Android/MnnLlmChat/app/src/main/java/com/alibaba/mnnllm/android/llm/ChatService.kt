// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.llm

import android.text.TextUtils
import com.alibaba.mnnllm.android.chat.model.ChatDataItem
import com.alibaba.mnnllm.android.model.ModelTypeUtils

class ChatService {
    private val transformerSessionMap: MutableMap<String, ChatSession> = HashMap()
    private val diffusionSessionMap: MutableMap<String, ChatSession> = HashMap()

    /**
     * Unified method to create a session for any model type
     * @param modelId The model ID
     * @param modelName The model name (used for type detection)
     * @param sessionIdParam Optional session ID, will generate new one if null/empty
     * @param historyList Optional chat history data
     * @param configPath Configuration file path for LLM models, or diffusion directory for diffusion models
     * @param useNewConfig If true, ignore existing config and use provided configPath. If false, may reuse existing session config
     * @param useCustomConfig If true, merge custom_config.json over base config (ChatActivity path). If false, use only base config (API/debug path).
     */
    @Synchronized
    fun createSession(
        modelId: String,
        modelName: String,
        sessionIdParam: String?,
        historyList: List<ChatDataItem>?,
        configPath: String?,
        useNewConfig: Boolean = false,
        useCustomConfig: Boolean = true
    ): ChatSession {
        val sessionId = if (TextUtils.isEmpty(sessionIdParam)) {
            System.currentTimeMillis().toString()
        } else {
            sessionIdParam!!
        }
        
        val session = if (ModelTypeUtils.isSanaModel(modelName)) {
            SanaSession(modelId, sessionId, configPath!!, historyList)
        } else if (ModelTypeUtils.isDiffusionModel(modelName)) {
            DiffusionSession(modelId, sessionId, configPath!!, historyList)
        } else {
            val llmSession = LlmSession(modelId, sessionId, configPath!!, historyList, useCustomConfig = useCustomConfig)
            llmSession.supportOmni = ModelTypeUtils.isOmni(modelName)
            llmSession
        }
        
        // Store in appropriate map
        if (session is LlmSession) {
            transformerSessionMap[sessionId] = session
        } else {
            diffusionSessionMap[sessionId] = session
        }
        
        return session
    }

    @Synchronized
    fun createLlmSession(
        modelId: String?,
        modelDir: String?,
        sessionIdParam: String?,
        chatDataItemList: List<ChatDataItem>?,
        supportOmni:Boolean,
        backendType: String? = null,
        useCustomConfig: Boolean = true
    ): LlmSession {
        var sessionId:String = if (TextUtils.isEmpty(sessionIdParam)) {
            System.currentTimeMillis().toString()
        } else {
            sessionIdParam!!
        }
        val session = LlmSession(modelId!!, sessionId, modelDir!!, chatDataItemList, backendType, useCustomConfig)
        session.supportOmni = supportOmni
        transformerSessionMap[sessionId] = session
        return session
    }

    @Synchronized
    fun createDiffusionSession(
        modelId: String?,
        modelDir: String?,
        sessionIdParam: String?,
        chatDataItemList: List<ChatDataItem>?
    ): ChatSession {
        var sessionId:String = if (TextUtils.isEmpty(sessionIdParam)) {
            System.currentTimeMillis().toString()
        } else {
            sessionIdParam!!
        }
        val session = DiffusionSession(modelId!!, sessionId, modelDir!!, chatDataItemList)
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
