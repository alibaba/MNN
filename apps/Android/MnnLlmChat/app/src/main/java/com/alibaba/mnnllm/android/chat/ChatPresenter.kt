// Created by ruoyi.sjd on 2025/5/6.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.chat

import android.text.TextUtils
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.chat.agent.AgentConsentController
import com.alibaba.mnnllm.android.chat.agent.AgentInference
import com.alibaba.mnnllm.android.chat.agent.AgentMessage
import com.alibaba.mnnllm.android.chat.agent.AgentPhase
import com.alibaba.mnnllm.android.chat.agent.AgentRequest
import com.alibaba.mnnllm.android.chat.agent.AgentRole
import com.alibaba.mnnllm.android.chat.agent.AgentRunner
import com.alibaba.mnnllm.android.chat.agent.AgentRunState
import com.alibaba.mnnllm.android.chat.agent.AgentStore
import com.alibaba.mnnllm.android.chat.agent.AgentStoredStep
import com.alibaba.mnnllm.android.chat.agent.AgentToolStepRecord
import java.time.Instant
import com.alibaba.mnnllm.android.chat.agent.AgentStreamDecoder
import com.alibaba.mnnllm.android.chat.agent.AgentToolExecutor
import com.alibaba.mnnllm.android.chat.agent.AgentTools
import com.alibaba.mnnllm.android.chat.agent.LlmSessionAgentInference
import com.alibaba.mnnllm.android.chat.agent.SqliteAgentStore
import com.alibaba.mnnllm.android.llm.LlmSession
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.sync.withLock
import java.util.UUID
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
import kotlinx.coroutines.CoroutineDispatcher
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
    private val modelId: String,
    val agentChat: AgentChatIntegration = AgentChatIntegration(
        storeFactory = { consent -> SqliteAgentStore(chatActivity.applicationContext, consent) }
    ),
    private val sessionFactory: ((Boolean, String?, List<ChatDataItem>?, Boolean) -> ChatSession)? = null,
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
    @Volatile
    private var loadingSession: ChatSession? = null
    @Volatile
    private var sessionUnavailable = false
    private var restoredHistory: List<ChatDataItem>? = null
    private val persistenceMutex = Mutex()
    private var persistedAgentTurn: Pair<String, AgentTurnPersistence>? = null
    private val additionalListeners = mutableListOf<GenerateListener>()
    @Volatile
    private var ownedAgentSession: ChatSession? = null
    @Volatile
    private var agentGeneration: Job? = null
    val isAgentMode: Boolean get() = agentChat.mode.value.agent

    suspend fun restoreAgentMode() {
        val id = chatActivity.intent.getStringExtra("chatSessionId")
        val history = withContext(Dispatchers.IO) {
            id?.let { chatDataManager!!.getChatDataBySession(it) }.orEmpty()
        }
        agentChat.restore(id, history)
    }

    suspend fun restoreCreatedSessionMode(): ChatSession = withContext(Dispatchers.IO) {
        generationMutex.withLock {
            check(!destroyed)
            val session = chatSession
            val id = checkNotNull(session.sessionId)
            val history = mergeHistory(session.getHistory().orEmpty(), chatDataManager!!.getChatDataBySession(id))
            sessionId = id
            restoredHistory = history
            sessionName = history.firstOrNull { it.type == ChatViewHolders.USER }?.text
            agentChat.restore(id, history)
            if (isAgentMode && session !== ownedAgentSession) {
                sessionUnavailable = true
                releaseOwnedSession(session)
                chatSession = createAgentSession(id, history)
                chatSession.setKeepHistory(true)
            }
            if (destroyed) {
                releaseOwnedSession(chatSession)
                throw CancellationException("ChatPresenter destroyed")
            }
            chatSession
        }
    }

    fun getChatHistory(): List<ChatDataItem> = restoredHistory ?: if (::chatSession.isInitialized)
        chatSession.getHistory().orEmpty() else emptyList()

    private fun mergeHistory(cached: List<ChatDataItem>, stored: List<ChatDataItem>): List<ChatDataItem> {
        val storedIds = stored.map { it.dbId }.filter { it >= 0 }.toSet()
        val storedItems = stored.toSet()
        return cached.filter { item ->
            if (item.dbId >= 0) item.dbId !in storedIds else item !in storedItems
        } + stored
    }

    suspend fun confirmAgentConsent() {
        check(!destroyed)
        agentChat.confirm()
    }

    suspend fun changeAgentMode(enabled: Boolean): ChatSession = withContext(Dispatchers.IO) {
        check(!destroyed && loadJob?.isActive != true && generationMutex.tryLock())
        try {
            val id = checkNotNull(chatSession.sessionId)
            val history = mergeHistory(restoredHistory ?: chatSession.getHistory().orEmpty(), chatDataManager!!.getChatDataBySession(id))
            if (history.any { it.type == ChatViewHolders.USER }) agentChat.lockMode()
            check(agentChat.mode.value.canChange)
            sessionId = id
            agentChat.changeMode(id, enabled)
            sessionUnavailable = true
            releaseOwnedSession(chatSession)
            chatSession = if (enabled) createAgentSession(id, history) else ensureOrdinarySession(id, history,
                chatActivity.intent.getStringExtra("configFilePath"), forceReload = true)
            sessionId = checkNotNull(chatSession.sessionId)
            restoredHistory = mergeHistory(chatSession.getHistory().orEmpty(), history)
            if (!enabled) agentChat.restore(sessionId, restoredHistory.orEmpty())
            check(enabled || !isAgentMode) { "An Agent session requires its own model session" }
            chatSession.setKeepHistory(true)
            if (destroyed) {
                releaseOwnedSession(chatSession)
                throw CancellationException("ChatPresenter destroyed")
            }
            chatSession
        } catch (e: Exception) {
            sessionUnavailable = true
            throw e
        } catch (e: OutOfMemoryError) {
            sessionUnavailable = true
            throw e
        } finally { generationMutex.unlock() }
    }

    private fun createAgentSession(id: String, history: List<ChatDataItem>): ChatSession {
        val session = sessionFactory?.invoke(true, id, history, true) ?: run {
            val path = checkNotNull(chatActivity.intent.getStringExtra("configFilePath"))
            LlmSession(chatActivity.modelId!!, id, path, history)
        }
        check(session.sessionId == id) { "Agent session identity changed" }
        return session.also { ownedAgentSession = it }
    }

    private fun ensureOrdinarySession(id: String?, history: List<ChatDataItem>?, configPath: String?,
        forceReload: Boolean = false): ChatSession {
        sessionFactory?.let { return it(false, id, history, forceReload) }
        val result = ServiceLocator.getLlmRuntimeController().ensureSession(
            modelId = chatActivity.modelId ?: modelId, forceReload = forceReload, useAppConfig = true,
            configPath = configPath, sessionId = id, historyList = history, deferLoad = true, owner = sessionOwner
        )
        check(result.success) { result.reason ?: "Failed to ensure LLM session" }
        return checkNotNull(result.session) { result.reason ?: "Session is not available" }
    }

    suspend fun retryAgentSession(): ChatSession = withContext(Dispatchers.IO) {
        generationMutex.withLock {
            check(isAgentMode && !destroyed && loadJob?.isActive != true)
            if (::chatSession.isInitialized) releaseOwnedSession(chatSession)
            sessionUnavailable = true
            val history = mergeHistory(restoredHistory.orEmpty(), chatDataManager!!.getChatDataBySession(sessionId!!))
            restoredHistory = history
            chatSession = createAgentSession(sessionId!!, history)
            chatSession.setKeepHistory(true)
            if (destroyed) {
                releaseOwnedSession(chatSession)
                throw CancellationException("ChatPresenter destroyed")
            }
            chatSession
        }
    }
    
    /**
     * Get LLM session instance
     * Provides safe access to the api.openai module
     * @return LlmSession instance, returns null if chatSession is not initialized or not of LlmSession type
     */
    fun getLlmSession(): com.alibaba.mnnllm.android.llm.LlmSession? {
        return if (!isAgentMode && ::chatSession.isInitialized && chatSession is com.alibaba.mnnllm.android.llm.LlmSession) {
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
        if (isAgentMode) return
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
        
        if (isAgentMode) {
            chatSession = createAgentSession(sessionId ?: UUID.randomUUID().toString(), chatDataItemList.orEmpty())
        } else if (ModelTypeUtils.isDiffusionModel(modelName) || ModelTypeUtils.isSanaModel(modelName)) {
            val chatService = ChatService.provide()
            chatSession = chatService.createSession(
                modelId, modelName, sessionId, chatDataItemList, configPath, false
            )
        } else {
            chatSession = ensureOrdinarySession(sessionId, chatDataItemList, configPath)
        }
        if (destroyed) {
            releaseOwnedSession(chatSession)
            throw CancellationException("ChatPresenter destroyed")
        }
        sessionId = chatSession.sessionId
        val actualHistory = mergeHistory(chatSession.getHistory().orEmpty(),
            chatDataManager!!.getChatDataBySession(checkNotNull(sessionId)))
        restoredHistory = actualHistory
        if (actualHistory.any { it.type == ChatViewHolders.USER }) agentChat.lockMode()
        sessionName = actualHistory.firstOrNull { it.type == ChatViewHolders.USER }?.text
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
                    if (!isAgentMode) (session as? ContextAwareSession)?.let {
                        contextManager.apply(it, contextManager.restore(session.sessionId, session.getHistory()))
                    }
                    loadOwnedSession(session)
                }
                currentCoroutineContext().ensureActive()
                check(session !is com.alibaba.mnnllm.android.llm.LlmSession || session.isModelLoaded()) { "Session was released during loading" }
                if (!isAgentMode) (session as? ContextAwareSession)?.let { contextManager.refresh(it) }
                withContext(Dispatchers.Main) {
                    if (!destroyed && chatSession === session) {
                        sessionUnavailable = false
                        chatActivity.onLoadingChanged(false)
                    }
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

    private suspend fun loadOwnedSession(session: ChatSession) {
        currentCoroutineContext().ensureActive()
        loadingSession = session
        try {
            if (destroyed) {
                releaseOwnedSession(session)
                throw CancellationException("ChatPresenter destroyed")
            }
            session.load()
        } finally {
            if (loadingSession === session) loadingSession = null
        }
    }

    private fun reportLoadFailure(session: ChatSession, message: String) {
        if (::chatSession.isInitialized && chatSession === session) sessionUnavailable = true
        chatActivity.lifecycleScope.launch {
            if (!destroyed && chatSession === session) chatActivity.onModelLoadFailed(message)
        }
        releaseOwnedSession(session)
    }

    private fun releaseOwnedSession(session: ChatSession) {
        if (session === ownedAgentSession) {
            session.release()
        } else if (session is com.alibaba.mnnllm.android.llm.LlmSession) {
            ServiceLocator.getLlmRuntimeController().releaseSession(session, sessionOwner)
        } else {
            session.release()
        }
    }

    fun reset(onResetSuccess: (newSessionId: String) -> Unit) {
        if (destroyed || isAgentMode || !::chatSession.isInitialized || loadJob?.isActive == true) return
        presenterScope.launch {
            if (!generationMutex.tryLock()) return@launch
            try {
                sessionId = chatSession.reset()
                (chatSession as? LlmSession)?.savedHistory = emptyList()
                restoredHistory = emptyList()
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

    private suspend fun submitAgentRequest(userData: ChatDataItem, history: List<ChatDataItem>): HashMap<String, Any> {
        val session = checkNotNull(ownedAgentSession as? LlmSession)
        check(session === chatSession && session.isModelLoaded()) { "Agent model is not available" }
        val config = session.captureAgentConfig()
        val limits = session.contextLimits()
        val limit = limits.limit.takeIf { it >= 256 } ?: ModelConfig.DEFAULT_CONTEXT_TOKEN_LIMIT
        val inference = LlmSessionAgentInference(session, preparePass = { pass ->
            check(pass.messages.all { AgentChatIntegration.isNativeText(it.content) }) { "Agent supports text only" }
            session.configureAgentPass(config, pass.outputTokenLimit, pass.allowThinking)
        })
        try {
            val state = coroutineScope {
                val turn = async {
                    if (stopGenerating) throw CancellationException("Agent stopped")
                    agentChat.run(inference, AgentRequest(
                        userText = userData.text.orEmpty(), history = AgentChatIntegration.history(history),
                        baseSystemPrompt = session.getSystemPrompt().orEmpty(), contextLimitTokens = limit,
                        outputReserveTokens = config.maxNewTokens, expectsThinking = config.expectsThinking, sessionId = sessionId
                    ))
                }
                agentGeneration = turn
                if (stopGenerating) turn.cancel()
                try { turn.await() }
                catch (e: CancellationException) {
                    currentCoroutineContext().ensureActive()
                    agentChat.state.value.copy(phase = AgentPhase.CANCELLED)
                }
            }
            return hashMapOf("agent" to true, "agent_state" to state,
                "response" to GenerateResultProcessor.answerDisplayText(state.answer),
                "error" to (state.phase == AgentPhase.FAILED), "session_invalid" to !session.isModelLoaded(),
                "message" to (state.error ?: ""))
        } finally {
            agentGeneration = null
            session.restoreAgentConfig(config)
        }
    }

    internal suspend fun persistAgentTurn(snapshot: AgentTurnSnapshot): AgentTurnPersistence = withContext(Dispatchers.IO) {
        persistenceMutex.withLock {
            val key = snapshot.sessionId + ":" + snapshot.turnTime
            persistedAgentTurn?.takeIf { it.first == key }?.let { return@withLock it.second }
            val state = snapshot.state
            var assistantId = -1L
            var failed = false
            if (state.phase == AgentPhase.COMPLETED && GenerateResultProcessor.answerDisplayText(state.answer).isNotBlank()) {
                val item = ChatDataItem(snapshot.messageTime, ChatViewHolders.ASSISTANT, state.answer).apply {
                    displayText = GenerateResultProcessor.answerDisplayText(state.answer)
                    benchmarkInfo = AgentChatIntegration.metrics(chatActivity, state)
                }
                try {
                    chatDataManager!!.addChatData(snapshot.sessionId, item)
                    assistantId = item.dbId
                    failed = assistantId < 0
                    if (assistantId >= 0 && snapshot.sessionId == sessionId) {
                        restoredHistory = mergeHistory(restoredHistory.orEmpty(), listOf(item))
                    }
                } catch (e: CancellationException) { throw e }
                catch (_: Exception) { failed = true }
            }
            if (state.steps.isNotEmpty()) {
                val record = AgentToolStepRecord(assistantId, snapshot.turnTime, state.phase,
                    state.steps.map(AgentStoredStep::from))
                try {
                    if (!agentChat.store.saveToolSteps(snapshot.sessionId, record, snapshot.consentRevision)) failed = true
                } catch (e: CancellationException) { throw e }
                catch (_: Exception) { failed = true }
            }
            AgentTurnPersistence(assistantId, failed).also { persistedAgentTurn = key to it }
        }
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
        if (sessionUnavailable) return hashMapOf("error" to true, "session_unavailable" to true)
        if (isAgentMode) {
            val reason = when {
                !agentChat.mode.value.authorized -> R.string.agent_consent_required
                !AgentChatIntegration.isTextOnly(userData) -> R.string.agent_text_only
                userData.text.isNullOrBlank() -> R.string.agent_empty_request
                else -> null
            }
            if (reason != null) return hashMapOf("error" to true, "agent_rejected" to reason)
        }
        if (!generationMutex.tryLock()) return hashMapOf("error" to true, "message" to "Generation already in progress")
        val owner = currentCoroutineContext()[Job]!!
        activeGeneration = owner
        stopGenerating = false
        var started = false
        val turnTime = Instant.now().toString()
        val turnMessageTime = userData.time ?: turnTime
        val turnConsentRevision = agentChat.consent.state.value.revision
        val turnSessionId = sessionId
        try {
            if (destroyed) throw CancellationException("ChatPresenter destroyed")
            if (sessionUnavailable || (::chatSession.isInitialized && chatSession is LlmSession &&
                    !(chatSession as LlmSession).isModelLoaded())) {
                return hashMapOf("error" to true, "session_unavailable" to true)
            }
            val result = try {
                check(::chatSession.isInitialized) { "Session is not available" }
                val prompt = PromptUtils.generateUserPrompt(userData)
                val agentHistory = if (isAgentMode) withContext(Dispatchers.IO) {
                    mergeHistory(restoredHistory.orEmpty(), chatDataManager!!.getChatDataBySession(checkNotNull(turnSessionId)))
                } else emptyList()
                val session = if (isAgentMode) null else contextSession
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
                    agentChat.lockMode()
                }
                if (isAgentMode) agentChat.beginTurn()
                withContext(Dispatchers.Main) {
                    generateListener.onGenerateStart()
                    additionalListeners.forEach { it.onGenerateStart() }
                    started = true
                }
                withContext(Dispatchers.IO) {
                    if (isAgentMode) submitAgentRequest(userData, agentHistory)
                    else submitRequest(prompt, userData, generateListener, owner)
                }
            } catch (e: CancellationException) {
                throw e
            } catch (e: OutOfMemoryError) {
                withContext(NonCancellable + Dispatchers.IO) { releaseOwnedSession(chatSession) }
                hashMapOf<String, Any>("error" to true, "message" to "Not enough memory to generate; reload the model", "session_invalid" to true)
            } catch (e: Exception) {
                hashMapOf<String, Any>("error" to true, "message" to (e.message ?: "Generation failed")).also {
                    if (isAgentMode && (ownedAgentSession as? LlmSession)?.isModelLoaded() != true) it["session_invalid"] = true
                }
            }
            if (isAgentMode) result["agent"] = true
            if (isAgentMode && turnSessionId != null) {
                val state = (result["agent_state"] as? AgentRunState ?: AgentRunState(phase = AgentPhase.FAILED))
                    .let { it.copy(steps = it.steps.toList(), metrics = it.metrics.toMap()) }
                result["agent_state"] = state
                val snapshot = AgentTurnSnapshot(checkNotNull(turnSessionId), turnTime,
                    turnMessageTime, turnConsentRevision, state)
                val persisted = persistAgentTurn(snapshot)
                result["agent_assistant_id"] = persisted.assistantId
                result["agent_assistant_time"] = snapshot.messageTime
                if (persisted.failed) result["agent_persistence_failed"] = true
            }
            withContext(Dispatchers.Main) {
                if (destroyed) return@withContext
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
                    val result = hashMapOf<String, Any>("error" to true, "message" to "Generation cancelled")
                    if (isAgentMode) {
                        result["agent"] = true
                        result["agent_state"] = agentChat.state.value.copy(phase = AgentPhase.CANCELLED)
                    }
                    generateListener.onGenerateFinished(result)
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
        agentGeneration?.cancel(CancellationException("Agent stopped"))
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
        val generation = activeGeneration
        val loading = loadJob
        val loadingTarget = loadingSession
        generation?.cancel(CancellationException("ChatPresenter destroyed"))
        presenterScope.cancel("ChatPresenter destroy")
        if (loading != null && !loading.isCompleted && loadingTarget != null &&
            (generation == null || generation.isCompleted)) {
            releaseOwnedSession(loadingTarget)
        }
        CoroutineScope(Dispatchers.IO + SupervisorJob()).launch {
            generation?.join()
            loading?.join()
            generationMutex.withLock {
                if (::chatSession.isInitialized) {
                    val session = chatSession
                    if (session === ownedAgentSession || session !is LlmSession ||
                        !ServerEventManager.getInstance().isServerRunning()) {
                        releaseOwnedSession(session)
                    }
                }
                agentChat.close()
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
                currentCoroutineContext().ensureActive()
                (newSession as? ContextAwareSession)?.let { contextManager.apply(it, carriedSummary) }
                updateDatabaseForModelSwitch(oldSessionId, newModelItem.modelId!!)
                withContext(Dispatchers.Main) { onSessionCreated(newSession) }
                if (newSession !is com.alibaba.mnnllm.android.llm.LlmSession ||
                    !(newSession as com.alibaba.mnnllm.android.llm.LlmSession).isModelLoaded()) {
                    loadOwnedSession(newSession)
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
            restoredHistory = currentChatHistory.toList()
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
        restoredHistory = currentChatHistory.toList()
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

internal data class AgentTurnSnapshot(
    val sessionId: String,
    val turnTime: String,
    val messageTime: String,
    val consentRevision: Long,
    val state: AgentRunState,
)

internal data class AgentTurnPersistence(val assistantId: Long, val failed: Boolean)

data class AgentChatMode(val agent: Boolean = false, val authorized: Boolean = false, val locked: Boolean = false) {
    val canChange: Boolean get() = !locked
}

class AgentChatIntegration(
    private val storeFactory: (AgentConsentController) -> AgentStore,
    val consent: AgentConsentController = AgentConsentController(),
    private val dispatcher: CoroutineDispatcher = Dispatchers.Default,
) {
    private val storeDelegate = lazy { storeFactory(consent) }
    val store: AgentStore get() = storeDelegate.value
    private val mutableMode = MutableStateFlow(AgentChatMode())
    val mode: StateFlow<AgentChatMode> get() = mutableMode
    private val mutableState = MutableStateFlow(AgentRunState())
    val state: StateFlow<AgentRunState> get() = mutableState
    private val modeMutex = Mutex()

    suspend fun restore(id: String?, history: List<ChatDataItem>) = modeMutex.withLock {
        consent.disable()
        mutableMode.value = AgentChatMode(agent = id?.let { store.isAgentSession(it) } == true,
            locked = history.any { it.type == ChatViewHolders.USER })
    }

    suspend fun confirm() = modeMutex.withLock {
        consent.confirmEnable(networkAllowed = true, persistenceAllowed = true)
        mutableMode.value = mutableMode.value.copy(authorized = true)
    }

    suspend fun changeMode(id: String, enabled: Boolean) = modeMutex.withLock {
        check(mutableMode.value.canChange)
        if (enabled) consent.confirmEnable(networkAllowed = true, persistenceAllowed = true)
        try {
            check(store.setAgentSession(id, enabled, consent.state.value.revision))
            if (!enabled) consent.disable()
            mutableMode.value = AgentChatMode(enabled, enabled)
            mutableState.value = AgentRunState()
        } catch (e: Exception) {
            consent.disable()
            mutableMode.value = mutableMode.value.copy(authorized = false)
            throw e
        }
    }

    fun lockMode() { mutableMode.value = mutableMode.value.copy(locked = true) }

    fun beginTurn() { mutableState.value = AgentRunState(phase = AgentPhase.RUNNING) }

    fun unlockForNewConversation() {
        check(!mode.value.agent)
        mutableMode.value = AgentChatMode()
        mutableState.value = AgentRunState()
    }

    suspend fun run(inference: AgentInference, request: AgentRequest,
        tools: AgentToolExecutor = AgentTools(consent), onState: (AgentRunState) -> Unit = {}): AgentRunState {
        check(mode.value.agent && mode.value.authorized)
        lockMode()
        return AgentRunner(inference, tools, store, consent, dispatcher = dispatcher).run(request) {
            mutableState.value = it
            onState(it)
        }
    }

    suspend fun close() {
        consent.disable()
        mutableMode.value = mutableMode.value.copy(authorized = false)
        if (storeDelegate.isInitialized()) (store as? SqliteAgentStore)?.close()
    }

    companion object {
        fun metrics(context: android.content.Context, state: AgentRunState): String =
            context.getString(R.string.agent_turn_metrics, state.pass, state.toolCalls, state.contextTokens,
                if (state.contextTokensEstimated) "~" else "")

        fun isNativeText(text: String): Boolean = listOf("<img>", "<audio>", "<video>").none { it in text }

        fun isTextOnly(item: ChatDataItem): Boolean = item.imageUris.isNullOrEmpty() &&
            item.audioUri == null && item.videoUri == null && isNativeText(item.text.orEmpty())

        fun history(items: List<ChatDataItem>): List<AgentMessage> = items.mapNotNull { item ->
            if (!isTextOnly(item)) return@mapNotNull null
            val role = when (item.type) {
                ChatViewHolders.USER -> AgentRole.USER
                ChatViewHolders.ASSISTANT -> AgentRole.ASSISTANT
                else -> return@mapNotNull null
            }
            val text = if (role == AgentRole.USER) item.text.orEmpty() else {
                GenerateResultProcessor.answerDisplayText(
                    AgentStreamDecoder.decode(item.text.orEmpty(), false).answer)
            }
            text.takeIf { it.isNotBlank() }?.let { AgentMessage(role, it) }
        }
    }
}
