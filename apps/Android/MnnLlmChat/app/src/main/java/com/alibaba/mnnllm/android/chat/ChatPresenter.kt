// Created by ruoyi.sjd on 2025/5/6.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.chat

import android.text.TextUtils
import android.util.Log
import androidx.lifecycle.lifecycleScope
import com.alibaba.mls.api.ModelItem
import com.alibaba.mnnllm.android.llm.ChatService
import com.alibaba.mnnllm.android.llm.ChatSession
import com.alibaba.mnnllm.android.llm.ContextAction
import com.alibaba.mnnllm.android.llm.ContextAwareSession
import com.alibaba.mnnllm.android.llm.ContextCheck
import com.alibaba.mnnllm.api.openai.di.ServiceLocator
import com.alibaba.mnnllm.api.openai.manager.ServerEventManager
import com.alibaba.mnnllm.android.chat.model.ChatDataItem
import com.alibaba.mnnllm.android.chat.model.ChatDataManager
import com.alibaba.mnnllm.android.chat.chatlist.ChatViewHolders
import com.alibaba.mnnllm.android.llm.GenerateProgressListener
import com.alibaba.mnnllm.android.utils.FileUtils
import com.alibaba.mnnllm.android.model.ModelTypeUtils
import com.alibaba.mnnllm.android.model.ModelUtils
import com.alibaba.mnnllm.android.modelsettings.ModelConfig
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.Job
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
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
    @Volatile
    var stopGenerating = false
    private var sessionId: String? = null
    private var sessionName:String? = null
    private var chatDataManager: ChatDataManager? = null
    private lateinit var chatSession: ChatSession
    private val presenterScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val sessionOwner = Any()
    private val generationMutex = Mutex()
    @Volatile
    private var activeGeneration: Job? = null
    @Volatile
    private var destroyed = false
    private var loadJob: Job? = null
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

    private val contextManager = ChatContextManager(ChatDataContextStore(chatDataManager!!))

    /** Usage indicator state; safe to collect from the UI. */
    val contextUsage: StateFlow<ContextUsageState> get() = contextManager.usage
    val contextNotices: SharedFlow<ContextNotice> get() = contextManager.notices

    @Volatile
    private var pendingContextCheck: ContextCheck = ContextCheck.NONE

    /** Database row of the message being sent, or -1 when it could not be stored. */
    @Volatile
    private var pendingUserRowId: Long = -1L

    private val contextSession: ContextAwareSession?
        get() = if (::chatSession.isInitialized) chatSession as? ContextAwareSession else null

    /** Re-reads the usage indicator from native, e.g. after the context settings changed. */
    fun refreshContextUsage() {
        val session = contextSession ?: return
        presenterScope.launch { contextManager.refresh(session) }
    }

    fun createSession(): ChatSession {
        val intent = chatActivity.intent
        sessionId = intent.getStringExtra("chatSessionId")
        Log.d(TAG, "createSession: received chatSessionId from intent: $sessionId")
        val chatDataItemList: List<ChatDataItem>?
        if (!TextUtils.isEmpty(sessionId)) {
            chatDataItemList = chatDataManager!!.getChatDataBySession(sessionId!!)
            Log.d(TAG, "createSession: queried database, got ${chatDataItemList.size} items for sessionId=$sessionId")
            if (chatDataItemList.isNotEmpty()) {
                sessionName = chatDataItemList[0].text
                Log.d(TAG, "createSession: first item text (sessionName): $sessionName")
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
        
        if (ModelTypeUtils.isDiffusionModel(modelName) || ModelTypeUtils.isSanaModel(modelName)) {
            val chatService = ChatService.provide()
            chatSession = chatService.createSession(
                modelId, modelName, sessionId, chatDataItemList, configPath, false
            )
        } else {
            val result = ServiceLocator.getLlmRuntimeController().ensureSession(
                modelId = modelId,
                forceReload = false,
                useAppConfig = true,
                configPath = configPath,
                sessionId = sessionId,
                historyList = chatDataItemList,
                deferLoad = true,
                owner = sessionOwner
            )
            if (!result.success || result.session == null) {
                throw IllegalStateException(result.reason ?: "Failed to ensure LLM session")
            }
            chatSession = result.session!!
        }
        if (destroyed) {
            releaseOwnedSession(chatSession)
            throw CancellationException("ChatPresenter destroyed")
        }
        sessionId = chatSession.sessionId
        chatSession.setKeepHistory(true)
        
        Log.d(TAG, "createSession: created session with sessionId=$sessionId, historySize=${chatSession.getHistory()?.size ?: 0}")
        return chatSession
    }

    fun load() {
        if (destroyed || loadJob?.isActive == true) return
        val session = chatSession
        loadJob = presenterScope.launch {
            try {
                withContext(Dispatchers.Main) { chatActivity.onLoadingChanged(true) }
                if (session !is com.alibaba.mnnllm.android.llm.LlmSession || !session.isModelLoaded()) {
                    (session as? ContextAwareSession)?.let {
                        contextManager.apply(it, contextManager.restore(session.sessionId, session.getHistory()))
                    }
                    session.load()
                }
                currentCoroutineContext().ensureActive()
                check(session !is com.alibaba.mnnllm.android.llm.LlmSession || session.isModelLoaded()) { "Session was released during loading" }
                (session as? ContextAwareSession)?.let { contextManager.refresh(it) }
                withContext(Dispatchers.Main) {
                    if (!destroyed && chatSession === session) chatActivity.onLoadingChanged(false)
                }
            } catch (e: CancellationException) {
                releaseOwnedSession(session)
                throw e
            } catch (e: OutOfMemoryError) {
                reportLoadFailure(session, "Not enough memory to load the model")
            } catch (e: Exception) {
                reportLoadFailure(session, e.message ?: "Model load failed")
            }
        }
    }

    private fun reportLoadFailure(session: ChatSession, message: String) {
        chatActivity.lifecycleScope.launch {
            if (!destroyed && chatSession === session) chatActivity.onModelLoadFailed(message)
        }
        releaseOwnedSession(session)
    }

    private fun releaseOwnedSession(session: ChatSession) {
        if (session is com.alibaba.mnnllm.android.llm.LlmSession) {
            ServiceLocator.getLlmRuntimeController().releaseSession(session, sessionOwner)
        } else {
            session.release()
        }
    }

    fun reset(onResetSuccess: (newSessionId: String) -> Unit) {
        if (destroyed || !::chatSession.isInitialized || loadJob?.isActive == true) return
        presenterScope.launch {
            if (!generationMutex.tryLock()) return@launch
            try {
                sessionId = chatSession.reset()
                contextSession?.let { session ->
                    session.contextSummary = null
                    session.contextAnchorId = -1L
                }
                contextManager.onConversationReset()
                sessionName = null
                withContext(Dispatchers.Main) { onResetSuccess(sessionId!!) }
            } catch (e: CancellationException) {
                throw e
            } catch (e: OutOfMemoryError) {
                reportLoadFailure(chatSession, "Not enough memory to reset the session")
            } catch (e: Exception) {
                reportLoadFailure(chatSession, e.message ?: "Session reset failed")
            } finally {
                generationMutex.unlock()
            }
        }
    }

    private fun submitDiffusionRequest(
        input: String, userData: ChatDataItem, listener: GenerateListener, owner: Job
    ): HashMap<String, Any> {
        val prompt = resolveDiffusionPrompt(input, modelId)
        val diffusionDestPath = FileUtils.generateDestDiffusionFilePath(
            chatActivity,
            sessionId!!
        )
        val imageInputPath = userData.imageUris?.firstOrNull()?.let { 
            FileUtils.getPathForUri(it)
        } ?: ""

        val config = ModelConfig.loadConfig(modelId)
        val steps = config?.diffusionSteps ?: ModelConfig.defaultConfig.diffusionSteps ?: 20
        val seed = if (config?.diffusionSeed != null && config.diffusionSeed!! != -1L) {
            config.diffusionSeed!!.toInt()
        } else {
            Random(System.currentTimeMillis()).nextInt()
        }
        val cfgPrompt = config?.cfgPrompt ?: "Generate high quality image"
        
        return chatSession.generate(
            prompt,
            mapOf(
                "output" to diffusionDestPath,
                "imageInput" to imageInputPath,
                "iterNum" to steps,
                "randomSeed" to seed,
                "cfgPrompt" to cfgPrompt
            )
            , object : GenerateProgressListener {
                override fun onProgress(progress: String?): Boolean {
                    chatActivity.lifecycleScope.launch {
                        if (activeGeneration === owner && owner.isActive && !destroyed) {
                            listener.onDiffusionGenerateProgress(progress, diffusionDestPath)
                            additionalListeners.forEach { it.onDiffusionGenerateProgress(progress, diffusionDestPath) }
                        }
                    }
                    return stopGenerating || !owner.isActive
                }
            }
        )
    }

    private fun submitLlmRequest(prompt: String, listener: GenerateListener, owner: Job): HashMap<String, Any> {
        contextSession?.let { session ->
            contextManager.compressIfNeeded(session, sessionId, pendingContextCheck, pendingUserRowId)
        }
        pendingContextCheck = ContextCheck.NONE
        owner.ensureActive()
        if (stopGenerating) {
            // Stopped while older turns were being summarized: nothing was generated.
            return hashMapOf("response" to "")
        }
        val generateResultProcessor =
            GenerateResultProcessor()
        generateResultProcessor.generateBegin()
        val result = chatSession.generate(prompt, mapOf(), object: GenerateProgressListener {
            override fun onProgress(progress: String?): Boolean {
                generateResultProcessor.process(progress)
                chatActivity.lifecycleScope.launch {
                    if (activeGeneration === owner && owner.isActive && !destroyed) {
                        listener.onLlmGenerateProgress(progress, generateResultProcessor)
                        additionalListeners.forEach { it.onLlmGenerateProgress(progress, generateResultProcessor) }
                    }
                }
                if (stopGenerating) {
                    Log.d(TAG, "stopGenerating requested")
                }
                return stopGenerating || !owner.isActive
            }
        })
        result["response"] = generateResultProcessor.getRawResult()
        return result
    }

    private fun submitRequest(
        input: String, userData: ChatDataItem, listener: GenerateListener, owner: Job
    ): HashMap<String, Any> {
        owner.ensureActive()
        val result = if (ModelTypeUtils.isDiffusionModel(modelName) || ModelTypeUtils.isSanaModel(modelName)) {
            submitDiffusionRequest(input, userData, listener, owner)
        } else {
            submitLlmRequest(input, listener, owner).also { contextManager.onTurnFinished(it, pendingUserRowId) }
        }
        if (result["session_invalid"] == true) releaseOwnedSession(chatSession)
        if (result["message"] == CONTEXT_TOO_LONG && userData.dbId >= 0) {
            chatDataManager?.deleteChatData(userData.dbId)
        }
        return result
    }

    private fun updateSession(sessionId: String, modelId: String?, sessionName: String) {
        chatDataManager!!.addOrUpdateSession(sessionId, modelId)
        chatDataManager!!.updateSessionName(this.sessionId!!, this.sessionName)
    }


    suspend fun requestGenerate(userData: ChatDataItem, generateListener: GenerateListener): HashMap<String, Any> {
        if (!generationMutex.tryLock()) return hashMapOf("error" to true, "message" to "Generation already in progress")
        val owner = currentCoroutineContext()[Job]!!
        activeGeneration = owner
        stopGenerating = false
        var started = false
        try {
            if (destroyed) throw CancellationException("ChatPresenter destroyed")
            val result = try {
                check(::chatSession.isInitialized) { "Session is not available" }
                val prompt = PromptUtils.generateUserPrompt(userData)
                val session = contextSession
                val check = if (session != null) {
                    withContext(Dispatchers.IO) { session.checkContext(prompt) }
                } else {
                    ContextCheck.NONE
                }
                if (check.action == ContextAction.TOO_LONG) {
                    withContext(Dispatchers.Main) {
                        generateListener.onContextRejected(userData)
                        additionalListeners.forEach { it.onContextRejected(userData) }
                    }
                    return hashMapOf("error" to true, "message" to CONTEXT_TOO_LONG, "context_rejected" to true)
                }
                pendingContextCheck = check
                withContext(Dispatchers.IO) {
                    if (sessionName.isNullOrEmpty()) {
                        sessionName = SessionUtils.generateSessionName(userData)
                        updateSession(sessionId!!, modelId, sessionName!!)
                    }
                    chatDataManager!!.addChatData(sessionId, userData)
                    pendingUserRowId = userData.dbId
                }
                withContext(Dispatchers.Main) {
                    generateListener.onGenerateStart()
                    additionalListeners.forEach { it.onGenerateStart() }
                    started = true
                }
                withContext(Dispatchers.IO) { submitRequest(prompt, userData, generateListener, owner) }
            } catch (e: CancellationException) {
                throw e
            } catch (e: OutOfMemoryError) {
                withContext(NonCancellable + Dispatchers.IO) { releaseOwnedSession(chatSession) }
                hashMapOf<String, Any>("error" to true, "message" to "Not enough memory to generate; reload the model", "session_invalid" to true)
            } catch (e: Exception) {
                hashMapOf<String, Any>("error" to true, "message" to (e.message ?: "Generation failed"))
            }
            withContext(Dispatchers.Main) {
                if (!started) {
                    generateListener.onGenerateStart()
                    additionalListeners.forEach { it.onGenerateStart() }
                }
                generateListener.onGenerateFinished(result)
                additionalListeners.forEach { it.onGenerateFinished(result) }
            }
            return result
        } catch (e: CancellationException) {
            stopGenerating = true
            if (started && !destroyed) {
                withContext(NonCancellable + Dispatchers.Main) {
                    generateListener.onGenerateFinished(hashMapOf("error" to true, "message" to "Generation cancelled"))
                }
            }
            throw e
        } finally {
            activeGeneration = null
            generationMutex.unlock()
        }
    }

    fun stopGenerate() {
        stopGenerating = true
        if (activeGeneration != null) contextSession?.cancelContextWork()
    }
    
    /**
     * Default GenerateListener that handles ChatActivity UI updates
     * This ensures all UI callbacks are executed on the main thread
     */
    private inner class DefaultChatActivityListener(private val userData: ChatDataItem) : GenerateListener {
        override fun onGenerateStart() {
            chatActivity.onGenerateStart(userData)
        }

        override fun onLlmGenerateProgress(progress: String?, generateResultProcessor: GenerateResultProcessor) {
            chatActivity.onLlmGenerateProgress(progress, generateResultProcessor)
        }

        override fun onDiffusionGenerateProgress(progress: String?, diffusionDestPath: String?) {
            chatActivity.onDiffusionGenerateProgress(progress, diffusionDestPath)
        }

        override fun onGenerateFinished(benchMarkResult: HashMap<String, Any>) {
            chatActivity.onGenerateFinished(benchMarkResult)
        }

        override fun onContextRejected(userData: ChatDataItem) {
            chatActivity.onContextRejected(userData)
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
        destroyed = true
        stopGenerate()
        activeGeneration?.cancel(CancellationException("ChatPresenter destroyed"))
        presenterScope.cancel("ChatPresenter destroy")
        val session = if (::chatSession.isInitialized) chatSession else return
        CoroutineScope(Dispatchers.IO + SupervisorJob()).launch {
            if (session !is com.alibaba.mnnllm.android.llm.LlmSession ||
                !ServerEventManager.getInstance().isServerRunning()) {
                releaseOwnedSession(session)
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
        if (::chatSession.isInitialized) chatSession.setEnableAudioOutput(enable)
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
        if (destroyed || loadJob?.isActive == true || !generationMutex.tryLock()) return
        loadJob = presenterScope.launch {
            try {
                withContext(Dispatchers.Main) { chatActivity.onLoadingChanged(true) }
                val oldSessionId = sessionId
                // Read before the old session goes away so the new one keeps the summary.
                val carriedSummary = contextManager.restore(oldSessionId, currentChatHistory)
                destroyCurrentSession()
                val newSession = createNewModelSession(newModelItem, currentChatHistory)
                (newSession as? ContextAwareSession)?.let { contextManager.apply(it, carriedSummary) }
                updateDatabaseForModelSwitch(oldSessionId, newModelItem.modelId!!)
                withContext(Dispatchers.Main) { onSessionCreated(newSession) }
                if (newSession !is com.alibaba.mnnllm.android.llm.LlmSession ||
                    !(newSession as com.alibaba.mnnllm.android.llm.LlmSession).isModelLoaded()) {
                    newSession.load()
                }
                (newSession as? ContextAwareSession)?.let { contextManager.refresh(it) }
                withContext(Dispatchers.Main) {
                    onSwitchComplete(newSession)
                    chatActivity.onLoadingChanged(false)
                }
            } catch (e: CancellationException) {
                destroyCurrentSession()
                throw e
            } catch (e: OutOfMemoryError) {
                reportLoadFailure(chatSession, "Not enough memory to switch models")
            } catch (e: Exception) {
                chatActivity.lifecycleScope.launch {
                    chatActivity.onLoadingChanged(false, loaded = false)
                    onSwitchError(e)
                }
                destroyCurrentSession()
            } finally {
                generationMutex.unlock()
            }
        }
    }
    
    private fun destroyCurrentSession() {
        if (::chatSession.isInitialized) releaseOwnedSession(chatSession)
    }
    
    private fun createNewModelSession(newModelItem: ModelItem, currentChatHistory: List<ChatDataItem>): ChatSession {
        val newModelId = newModelItem.modelId!!
        val newModelName = newModelItem.modelName!!
        if (ModelTypeUtils.isDiffusionModel(newModelName) || ModelTypeUtils.isSanaModel(newModelName)) {
            val chatService = ChatService.provide()
            val newConfigPath = ModelUtils.getConfigPathForModel(newModelItem)
            val newSession = chatService.createSession(
                newModelId, newModelName,
                null, currentChatHistory,
                newConfigPath, true
            )
            chatSession = newSession
            sessionId = newSession.sessionId
            chatSession.setKeepHistory(true)
            return newSession
        }
        val newConfigPath = ModelUtils.getConfigPathForModel(newModelItem)
        val result = ServiceLocator.getLlmRuntimeController().ensureSession(
            modelId = newModelId,
            forceReload = true,
            useAppConfig = true,
            configPath = newConfigPath,
            sessionId = null,
            historyList = currentChatHistory,
            deferLoad = true,
            owner = sessionOwner
        )
        if (!result.success || result.session == null) {
            throw IllegalStateException(result.reason ?: "Failed to ensure LLM session for model switch")
        }
        val newSession = result.session!!
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
    
    companion object {
        private const val TAG: String = "ChatPresenter"
        const val CONTEXT_TOO_LONG = "context_too_long"
        internal fun resolveDiffusionPrompt(input: String, modelId: String): String {
            if (input.isNotBlank()) return input
            return if (ModelTypeUtils.requiresFaceImageInput(modelId)) {
                DEFAULT_SANA_PROMPT
            } else {
                "A cyberpunk cat in neon lights"
            }
        }
    }

    interface GenerateListener {
        fun onDiffusionGenerateProgress(progress: String?, diffusionDestPath: String?)
        fun onGenerateStart()
        fun onGenerateFinished(benchMarkResult: HashMap<String, Any>)
        fun onLlmGenerateProgress(progress: String?, generateResultProcessor: GenerateResultProcessor)
        /**
         * The message alone exceeds the context budget; nothing was stored or generated.
         * By default this finishes the request with an error so listeners that only wait
         * for [onGenerateFinished] (voice chat) do not hang.
         */
        fun onContextRejected(userData: ChatDataItem) {
            onGenerateFinished(hashMapOf("error" to true, "message" to CONTEXT_TOO_LONG, "context_rejected" to true))
        }
    }
}
