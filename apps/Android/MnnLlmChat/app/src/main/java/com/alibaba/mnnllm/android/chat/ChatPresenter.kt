// Created by ruoyi.sjd on 2025/5/6.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.chat

import android.text.TextUtils
import android.util.Log
import androidx.lifecycle.lifecycleScope
import com.alibaba.mls.api.ModelItem
import com.alibaba.mnnllm.android.llm.ChatService
import com.alibaba.mnnllm.android.llm.ChatSession
import com.alibaba.mnnllm.android.chat.model.ChatDataItem
import com.alibaba.mnnllm.android.chat.model.ChatDataManager
import com.alibaba.mnnllm.android.chat.chatlist.ChatViewHolders
import com.alibaba.mnnllm.android.llm.GenerateProgressListener
import com.alibaba.mnnllm.android.utils.FileUtils
import com.alibaba.mnnllm.android.model.ModelTypeUtils
import com.alibaba.mnnllm.android.model.ModelUtils
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.async
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import java.text.DateFormat
import java.util.Random

/**
 * ChatPresenter
 */
class ChatPresenter(
    private val chatActivity: ChatActivity,
    private val modelName: String,
    private val modelId: String
) {
    val dateFormat: DateFormat get() = chatActivity.dateFormat!!
    var stopGenerating = false
    private var sessionId: String? = null
    private var sessionName:String? = null
    private var chatDataManager: ChatDataManager? = null
    private lateinit var chatSession: ChatSession
    private val presenterScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private var generateListener:GenerateListener? = null
    private val additionalListeners = mutableListOf<GenerateListener>()
    
    /**
     * Get LLM session instance
     * Provides safe access to the api.openai module
     * @return LlmSession instance, returns null if chatSession is not initialized or not of LlmSession type
     */
    fun getLlmSession(): com.alibaba.mnnllm.android.llm.LlmSession? {
        return if (::chatSession.isInitialized && chatSession is com.alibaba.mnnllm.android.llm.LlmSession) {
            chatSession as com.alibaba.mnnllm.android.llm.LlmSession
        } else {
            null
        }
    }
    
    /**
     * Get current session ID
     * @return Session ID, returns null if not set
     */
    fun getSessionId(): String? {
        return sessionId
    }
    
    /**
     * Add an additional generate listener for multi-UI updates
     */
    fun addGenerateListener(listener: GenerateListener) {
        additionalListeners.add(listener)
    }
    
    /**
     * Remove an additional generate listener
     */
    fun removeGenerateListener(listener: GenerateListener) {
        additionalListeners.remove(listener)
    }

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
            Log.d(TAG, "createSession: loaded ${chatDataItemList.size} history items for sessionId=$sessionId")
        } else {
            chatDataItemList = null
            Log.d(TAG, "createSession: no sessionId provided, starting new session")
        }
        
        val configPath = if (ModelTypeUtils.isDiffusionModel(modelName)) {
            intent.getStringExtra("diffusionDir")
        } else {
            intent.getStringExtra("configFilePath")
        }
        
        Log.d(TAG, "createSession: isDiffusion=${ModelTypeUtils.isDiffusionModel(modelName)}, modelName=$modelName, configPath=$configPath")
        
        chatSession = chatService.createSession(
            modelId, modelName, sessionId, chatDataItemList, configPath, false
        )
        sessionId = chatSession.sessionId
        chatSession.setKeepHistory(true)
        
        Log.d(TAG, "createSession: created session with sessionId=$sessionId, historySize=${chatSession.getHistory()?.size ?: 0}")
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
                        additionalListeners.forEach { it.onDiffusionGenerateProgress(progress, diffusionDestPath) }
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
                    additionalListeners.forEach { it.onLlmGenerateProgress(progress, generateResultProcessor) }
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
        val benchMarkResult = try {
            if (ModelTypeUtils.isDiffusionModel(this.modelName)) {
                submitDiffusionRequest(input)
            } else {
                submitLlmRequest(input)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error during generation request", e)
            // Create a basic error result to ensure onGenerateFinished is called
            HashMap<String, Any>().apply {
                put("error", true)
                put("message", e.message ?: "Generation failed")
                put("response", "生成失败，请重试")
            }
        }
        
        chatActivity.lifecycleScope.launch {
            this@ChatPresenter.generateListener?.onGenerateFinished(benchMarkResult)
            additionalListeners.forEach { it.onGenerateFinished(benchMarkResult) }
        }
        return benchMarkResult
    }

    private fun updateSession(sessionId: String, modelId: String?, sessionName: String) {
        chatDataManager!!.addOrUpdateSession(sessionId, modelId)
        chatDataManager!!.updateSessionName(this.sessionId!!, this.sessionName)
    }


    suspend fun requestGenerate(userData: ChatDataItem, generateListener: GenerateListener): HashMap<String, Any> {
        this.generateListener = generateListener
        val prompt = PromptUtils.generateUserPrompt(userData)
        
        // Ensure user input is saved first
        try {
            if (this.sessionName.isNullOrEmpty()) {
                this.sessionName = SessionUtils.generateSessionName(userData)
                updateSession(sessionId!!, modelId, sessionName!!)
            }
            
            // Always save user input to database first
            Log.d(TAG, "requestGenerate: saving user input for sessionId=$sessionId")
            chatDataManager!!.addChatData(sessionId, userData)
            
            this.generateListener?.onGenerateStart()
            additionalListeners.forEach { it.onGenerateStart() }
            
            val result = presenterScope.async {
                return@async submitRequest(prompt, userData)
            }.await()
            
            return result
        } catch (e: Exception) {
            Log.e(TAG, "requestGenerate: Error during request generation", e)
            
            // Still try to save user input even if generation fails
            try {
                if (sessionId != null) {
                    chatDataManager!!.addChatData(sessionId, userData)
                }
            } catch (saveException: Exception) {
                Log.e(TAG, "requestGenerate: Failed to save user input", saveException)
            }
            
            // Return error result
            val errorResult = HashMap<String, Any>().apply {
                put("error", true)
                put("message", e.message ?: "Request generation failed")
                put("response", "生成失败，请重试")
            }
            
            // Still call onGenerateFinished to ensure UI is updated
            this.generateListener?.onGenerateFinished(errorResult)
            additionalListeners.forEach { it.onGenerateFinished(errorResult) }
            
            return errorResult
        }
    }

    fun stopGenerate() {
        stopGenerating = true
    }
    
    /**
     * Default GenerateListener that handles ChatActivity UI updates
     * This ensures all UI callbacks are executed on the main thread
     */
    private inner class DefaultChatActivityListener(private val userData: ChatDataItem) : GenerateListener {
        override fun onGenerateStart() {
            chatActivity.lifecycleScope.launch {
                chatActivity.onGenerateStart(userData)
            }
        }
        
        override fun onLlmGenerateProgress(progress: String?, generateResultProcessor: GenerateResultProcessor) {
            chatActivity.lifecycleScope.launch {
                chatActivity.onLlmGenerateProgress(progress, generateResultProcessor)
            }
        }
        
        override fun onDiffusionGenerateProgress(progress: String?, diffusionDestPath: String?) {
            chatActivity.lifecycleScope.launch {
                chatActivity.onDiffusionGenerateProgress(progress, diffusionDestPath)
            }
        }
        
        override fun onGenerateFinished(benchMarkResult: HashMap<String, Any>) {
            chatActivity.lifecycleScope.launch {
                chatActivity.onGenerateFinished(benchMarkResult)
            }
        }
    }
    
    /**
     * Send a text message - unified method for both regular and voice messages
     * This ensures proper session management and database storage
     */
    suspend fun sendMessage(text: String): HashMap<String, Any> {
        val userData = ChatDataItem(ChatViewHolders.USER)
        userData.text = text
        userData.time = dateFormat.format(java.util.Date())
        return requestGenerate(userData, DefaultChatActivityListener(userData))
    }
    
    /**
     * Send a pre-created ChatDataItem - for more complex message types
     */
    suspend fun sendMessage(userData: ChatDataItem): HashMap<String, Any> {
        return requestGenerate(userData, DefaultChatActivityListener(userData))
    }

    fun destroy() {
        stopGenerate()
        presenterScope.cancel("ChatPresenter destroy")
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
        try {
            Log.d(TAG, "saveResponseToDatabase: saving response for sessionId=$sessionId")
            this.chatDataManager?.addChatData(sessionId, recentItem)
        } catch (e: Exception) {
            Log.e(TAG, "saveResponseToDatabase: Failed to save response to database for sessionId=$sessionId", e)
        }
    }

    fun setEnableAudioOutput(enable: Boolean) {
        this.chatSession.setEnableAudioOutput(enable)
    }

    /**
     * Switch to a new model while preserving chat history
     */
    fun switchModel(
        newModelItem: ModelItem,
        currentChatHistory: List<ChatDataItem>,
        onSwitchComplete: (ChatSession) -> Unit,
        onSwitchError: (Exception) -> Unit,
        onSessionCreated: (ChatSession) -> Unit
    ) {
        presenterScope.launch {
            try {
                chatActivity.lifecycleScope.launch {
                    chatActivity.onLoadingChanged(true)
                }
                val oldSessionId = sessionId
                destroyCurrentSession()
                val newSession = createNewModelSession(newModelItem)
                applyChatHistoryToNewSession(newSession, currentChatHistory)
                updateDatabaseForModelSwitch(oldSessionId, newModelItem.modelId!!)
                chatActivity.lifecycleScope.launch {
                    onSessionCreated(newSession)
                }
                newSession.load()
                chatActivity.lifecycleScope.launch {
                    onSwitchComplete(newSession)
                    chatActivity.onLoadingChanged(false)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error switching model", e)
                chatActivity.lifecycleScope.launch {
                    onSwitchError(e)
                    chatActivity.onLoadingChanged(false)
                }
            }
        }
    }
    
    private fun destroyCurrentSession() {
        if (::chatSession.isInitialized) {
            chatSession.reset()
            chatSession.release()
        }
    }
    
    private fun createNewModelSession(newModelItem: ModelItem): ChatSession {
        val chatService = ChatService.provide()
        val newModelId = newModelItem.modelId!!
        val newModelName = newModelItem.modelName!!
        val newConfigPath = ModelUtils.getConfigPathForModel(newModelItem)
        val newSession = chatService.createSession(
            newModelId, newModelName,
            null, null, // No previous session data
            newConfigPath, true // Use new config
        )
        chatSession = newSession
        sessionId = newSession.sessionId
        chatSession.setKeepHistory(true)
        return newSession
    }
    
    private fun updateDatabaseForModelSwitch(oldSessionId: String?, newModelId: String) {
        if (oldSessionId != null) {
            chatDataManager?.updateSessionModelId(oldSessionId, newModelId)
        }
    }
    
    private fun applyChatHistoryToNewSession(newSession: ChatSession, currentChatHistory: List<ChatDataItem>) {
        if (newSession is com.alibaba.mnnllm.android.llm.LlmSession && currentChatHistory.isNotEmpty()) {
            newSession.savedHistory = currentChatHistory
        }
    }

    companion object {
        private const val TAG: String = "ChatPresenter"
    }

    interface GenerateListener {
        fun onDiffusionGenerateProgress(progress: String?, diffusionDestPath: String?)
        fun onGenerateStart()
        fun onGenerateFinished(benchMarkResult: HashMap<String, Any>)
        fun onLlmGenerateProgress(progress: String?, generateResultProcessor: GenerateResultProcessor)
    }
}