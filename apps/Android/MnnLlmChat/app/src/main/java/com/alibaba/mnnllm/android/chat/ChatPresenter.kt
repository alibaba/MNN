// Created by ruoyi.sjd on 2025/5/6.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.chat

import android.text.TextUtils
import android.util.Log
import androidx.lifecycle.lifecycleScope
import com.alibaba.mls.api.ModelItem
import com.alibaba.mnnllm.android.llm.ChatService
import com.alibaba.mnnllm.android.llm.ChatSession
import com.alibaba.mnnllm.api.openai.di.ServiceLocator
import com.alibaba.mnnllm.api.openai.manager.ServerEventManager
import com.alibaba.mnnllm.android.agent.AgenticPrompts
import com.alibaba.mnnllm.android.agent.AgenticOutputParser
import com.alibaba.mnnllm.android.agent.AgenticToolExecutor
import com.alibaba.mnnllm.android.agent.AgentSystemCall
import com.alibaba.mnnllm.android.chat.model.ChatDataItem
import com.alibaba.mnnllm.android.chat.model.ChatFileAttachment
import com.alibaba.mnnllm.android.chat.model.ChatDataManager
import com.alibaba.mnnllm.android.chat.chatlist.ChatViewHolders
import com.alibaba.mnnllm.android.llm.GenerateProgressListener
import com.alibaba.mnnllm.android.utils.FileUtils
import com.alibaba.mnnllm.android.model.ModelTypeUtils
import com.alibaba.mnnllm.android.model.ModelUtils
import com.alibaba.mnnllm.android.modelsettings.ModelConfig
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.async
import kotlinx.coroutines.cancel
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
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
    private var agentEnabled: Boolean = false
    
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

    fun setAgentEnabled(enabled: Boolean) {
        agentEnabled = enabled
        sessionId?.let {
            chatDataManager?.updateSessionMode(
                it,
                if (enabled) {
                    com.alibaba.mnnllm.android.chat.model.ChatDatabaseHelper.SESSION_MODE_AGENT
                } else {
                    com.alibaba.mnnllm.android.chat.model.ChatDatabaseHelper.SESSION_MODE_NORMAL
                }
            )
        }
        applySystemPromptForCurrentMode()
    }

    fun isAgentEnabled(): Boolean {
        return agentEnabled
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
                deferLoad = true
            )
            if (!result.success || result.session == null) {
                throw IllegalStateException(result.reason ?: "Failed to ensure LLM session")
            }
            chatSession = result.session!!
        }
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
            try {
                if (chatSession is com.alibaba.mnnllm.android.llm.LlmSession &&
                    (chatSession as com.alibaba.mnnllm.android.llm.LlmSession).isModelLoaded()) {
                    Log.d(TAG, "chatSession already loaded by LlmRuntimeController, skipping load")
                } else {
                    chatSession.load()
                }
                applySystemPromptForCurrentMode()
                chatActivity.lifecycleScope.launch {
                    chatActivity.onLoadingChanged(false)
                }
                Log.d(TAG, "chatSession loaded")
            } catch (e: IllegalStateException) {
                Log.e(TAG, "Model load failed: ${e.message}", e)
                if (chatSession is com.alibaba.mnnllm.android.llm.LlmSession) {
                    ServiceLocator.getLlmRuntimeController().releaseSession()
                }
                chatActivity.lifecycleScope.launch {
                    chatActivity.onLoadingChanged(false)
                    chatActivity.onModelLoadFailed(e.message ?: "Model load failed")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Model load failed with unexpected error", e)
                if (chatSession is com.alibaba.mnnllm.android.llm.LlmSession) {
                    ServiceLocator.getLlmRuntimeController().releaseSession()
                }
                chatActivity.lifecycleScope.launch {
                    chatActivity.onLoadingChanged(false)
                    chatActivity.onModelLoadFailed(e.message ?: "Model load failed")
                }
            }
        }
    }

    fun reset(onResetSuccess: (newSessionId: String) -> Unit) {
        presenterScope.launch {
            // Don't delete chat data - preserve history for the old session
            // Just reset the session to get a new sessionId
            sessionId = chatSession.reset()
            sessionName = null  // Clear session name for the new session
            chatActivity.lifecycleScope.launch {
                onResetSuccess(sessionId!!)
            }
        }
    }

    private fun applySystemPromptForCurrentMode() {
        val llmSession = getLlmSession() ?: return
        val currentModelId = chatActivity.modelId ?: modelId
        val currentModelName = chatActivity.modelName.ifBlank { modelName }
        val systemPrompt = if (agentEnabled) {
            AgenticPrompts.buildSystemPromptForModel(
                modelName = currentModelName,
                modelId = currentModelId,
                memoryBlock = chatDataManager?.buildAgentMemoryBlock().orEmpty(),
                skillBlock = chatDataManager?.buildAgentSkillBlock().orEmpty()
            )
        } else {
            ModelConfig.loadConfig(currentModelId)?.systemPrompt
                ?: ModelConfig.defaultConfig.systemPrompt
                ?: "You are a helpful assistant."
        }
        llmSession.updateSystemPrompt(systemPrompt)
        Log.d(TAG, "Agent mode ${if (agentEnabled) "enabled" else "disabled"} for model=$currentModelId")
    }

    private fun submitDiffusionRequest(input: String, userData: ChatDataItem): HashMap<String, Any> {
        val currentModelId = chatActivity.modelId ?: modelId
        val prompt = resolveDiffusionPrompt(input, currentModelId)
        val diffusionDestPath = FileUtils.generateDestDiffusionFilePath(
            chatActivity,
            sessionId!!
        )
        val imageInputPath = userData.imageUris?.firstOrNull()?.let { 
            FileUtils.getPathForUri(it)
        } ?: ""

        val config = ModelConfig.loadConfig(currentModelId)
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
                        this@ChatPresenter.generateListener?.onDiffusionGenerateProgress(progress, diffusionDestPath)
                        additionalListeners.forEach { it.onDiffusionGenerateProgress(progress, diffusionDestPath) }
                    }
                    return false
                }
            }
        )
    }

    private fun submitLlmRequest(prompt:String, emitProgress: Boolean = true): HashMap<String, Any> {
        val generateResultProcessor =
            GenerateResultProcessor()
        generateResultProcessor.generateBegin()
        val result = chatSession.generate(prompt, mapOf(), object: GenerateProgressListener {
            override fun onProgress(progress: String?): Boolean {
                generateResultProcessor.process(progress)
                if (emitProgress) {
                    chatActivity.lifecycleScope.launch {
                        this@ChatPresenter.generateListener?.onLlmGenerateProgress(progress, generateResultProcessor)
                        additionalListeners.forEach { it.onLlmGenerateProgress(progress, generateResultProcessor) }
                    }
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

    private suspend fun submitAgenticLlmRequest(input: String): HashMap<String, Any> {
        val statusLog = StringBuilder()
        val searchedQueries = mutableSetOf<String>()
        val visitedUrls = mutableSetOf<String>()
        val executedPythonKeys = mutableSetOf<String>()
        var totalToolCalls = 0
        var totalBrowserCalls = 0
        var totalPythonCalls = 0
        val aggregateMetrics = HashMap<String, Any>()
        val generatedFiles = mutableListOf<ChatFileAttachment>()
        val budget = AgenticPrompts.defaultLoopBudget

        fun appendStatus(line: String) {
            statusLog.appendLine(line)
            emitAgentStatus(statusLog.toString().trimEnd())
        }

        fun ensureNotStopped() {
            if (stopGenerating) {
                throw kotlinx.coroutines.CancellationException("Agent generation stopped")
            }
        }

        appendStatus("[Agent] 规划中...")
        ensureNotStopped()
        applySystemPromptForCurrentMode()
        var llmResult = submitLlmRequest(input, emitProgress = false)
        mergeLlmMetrics(aggregateMetrics, llmResult)
        var raw = llmResult["response"] as? String ?: ""
        var parsed = AgenticOutputParser.parse(raw)

        for (pass in 1..budget.maxPasses) {
            currentCoroutineContext().ensureActive()
            ensureNotStopped()
            val calls = parsed?.systemCalls.orEmpty()
            if (calls.isEmpty()) break

            appendStatus("[Agent] 第 $pass 轮计划：${calls.size} 个工具调用")
            val executableCalls = mutableListOf<AgentSystemCall>()
            for (call in calls) {
                val type = call.type.orEmpty().trim().lowercase()
                val isSearch = type == "web_search"
                val isBrowse = type == "browser_url" || type == "browse_url" || type == "web_browse" || type == "open_url"
                val isPython = type == "python_exec" || type == "run_python" || type == "python"
                val key = when {
                    isSearch -> call.query.orEmpty().trim().lowercase()
                    isBrowse -> call.url.orEmpty().ifBlank { call.query.orEmpty() }.trim().lowercase()
                    isPython -> type + ":" + call.code.orEmpty().take(500) + ":" + call.input.orEmpty().take(500)
                    else -> type
                }
                val duplicate = (isSearch && key in searchedQueries) ||
                    (isBrowse && key in visitedUrls) ||
                    (isPython && key in executedPythonKeys)
                val budgetBlocked = totalToolCalls >= budget.maxToolCalls ||
                    (isBrowse && totalBrowserCalls >= budget.maxBrowserCalls) ||
                    (isPython && totalPythonCalls >= budget.maxPythonCalls)
                when {
                    duplicate -> appendStatus("[Agent] 跳过重复调用：$key")
                    budgetBlocked -> appendStatus("[Agent] 跳过工具调用：预算已用尽")
                    else -> {
                        if (isSearch) searchedQueries += key
                        if (isBrowse) {
                            visitedUrls += key
                            totalBrowserCalls += 1
                        }
                        if (isPython) {
                            executedPythonKeys += key
                            totalPythonCalls += 1
                        }
                        totalToolCalls += 1
                        executableCalls += call
                    }
                }
            }

            if (executableCalls.isEmpty()) break

            ensureNotStopped()
            val toolExecutionResult = AgenticToolExecutor.execute(executableCalls) { event ->
                when (event.type) {
                    AgenticToolExecutor.ToolStepEvent.Type.STARTED ->
                        appendStatus("[Agent] ${event.title}：${event.detail}")
                    AgenticToolExecutor.ToolStepEvent.Type.FINISHED ->
                        appendStatus("[Agent] ${event.title}完成：${event.detail}")
                    AgenticToolExecutor.ToolStepEvent.Type.FAILED ->
                        appendStatus("[Agent] ${event.title}失败：${event.detail}")
                }
            }
            generatedFiles.addAll(toolExecutionResult.generatedFiles)

            val remaining = budget.maxToolCalls - totalToolCalls
            appendStatus("[Agent] 已获得工具结果，继续推理...")
            ensureNotStopped()
            val continuationInput = buildString {
                appendLine("system_calls 执行结果：")
                appendLine(toolExecutionResult.observations)
                appendLine()
                if (remaining > 0) {
                    appendLine("Tool budget: remaining=$remaining. Continue from the existing conversation context and KV cache. Do not restate or re-send the original user question unless needed for the final answer. You may continue calling get_current_time, web_search, browser_url, or python_exec if useful. Prefer primary sources, use Python for calculation/data work, and avoid duplicate queries, URLs, or code. Return final reply when the answer is sufficiently supported.")
                } else {
                    appendLine("Tool budget: remaining=0. Do not call more tools. Continue from the existing conversation context and return the best final reply from the available results.")
                }
            }
            llmResult = submitLlmRequest(continuationInput, emitProgress = false)
            mergeLlmMetrics(aggregateMetrics, llmResult)
            raw = llmResult["response"] as? String ?: ""
            parsed = AgenticOutputParser.parse(raw)
        }

        if (parsed?.hasToolCalls() == true && parsed?.reply.isNullOrBlank()) {
            appendStatus("[Agent] 工具预算结束，整理最终回答...")
            ensureNotStopped()
            val finalOnlyPrompt = buildString {
                appendLine("Tool budget is exhausted. Do not call tools again. Continue from the existing conversation context and KV cache. Return the best final user-facing reply based on the observations already in this conversation. Do not expose JSON.")
            }
            llmResult = submitLlmRequest(finalOnlyPrompt, emitProgress = false)
            mergeLlmMetrics(aggregateMetrics, llmResult)
            raw = llmResult["response"] as? String ?: ""
            parsed = AgenticOutputParser.parse(raw)
        }

        persistAgentUpdates(parsed)
        val localSkillHints = chatDataManager?.runLocalAgentSkills(input).orEmpty()
        val baseReply = parsed?.reply?.takeIf { it.isNotBlank() } ?: raw
        val finalReply = if (localSkillHints.isBlank()) {
            baseReply
        } else {
            "$baseReply\n\n$localSkillHints"
        }
        generatedFiles.addAll(AgenticToolExecutor.resolveMentionedWorkspaceFiles(finalReply))
        applySystemPromptForCurrentMode()
        return HashMap<String, Any>().apply {
            putAll(aggregateMetrics)
            put("response", finalReply)
            put("replace_display", true)
            put("agent_steps", statusLog.toString().trimEnd())
            if (generatedFiles.isNotEmpty()) {
                put("generated_files", generatedFiles.distinctBy { it.path })
            }
        }
    }

    private fun mergeLlmMetrics(target: HashMap<String, Any>, source: HashMap<String, Any>) {
        listOf("input_len", "prompt_len", "decode_len", "prefill_time", "decode_time", "vision_time", "audio_time").forEach { key ->
            val sourceValue = metricAsLong(source[key])
            if (sourceValue > 0L) {
                target[key] = metricAsLong(target[key]) + sourceValue
            }
        }
    }

    private fun metricAsLong(value: Any?): Long {
        return when (value) {
            is Long -> value
            is Int -> value.toLong()
            is Number -> value.toLong()
            is String -> value.toLongOrNull() ?: 0L
            else -> 0L
        }
    }

    private fun persistAgentUpdates(parsed: com.alibaba.mnnllm.android.agent.AgenticResponse?) {
        val manager = chatDataManager ?: return
        parsed?.memoryUpdates.orEmpty().forEach { update ->
            manager.upsertAgentMemory(update.category, update.content, source = "agent")
        }
        parsed?.skillUpdates.orEmpty().forEach { update ->
            manager.upsertAgentSkill(
                name = update.name,
                description = update.description,
                triggerKeywords = update.triggerKeywords,
                actionTemplate = update.actionTemplate
            )
        }
    }

    private fun emitAgentStatus(status: String) {
        chatActivity.lifecycleScope.launch {
            this@ChatPresenter.generateListener?.onAgentStatus(status)
            additionalListeners.forEach { it.onAgentStatus(status) }
        }
    }

    private suspend fun submitRequest(input: String, userData: ChatDataItem): HashMap<String, Any> {
        stopGenerating = false
        val benchMarkResult = try {
            val currentModelName = chatActivity.modelName.ifBlank { modelName }
            if (ModelTypeUtils.isDiffusionModel(currentModelName)) {
                submitDiffusionRequest(input, userData)
            } else if (agentEnabled && getLlmSession() != null) {
                submitAgenticLlmRequest(input)
            } else {
                submitLlmRequest(input)
            }
        } catch (e: kotlinx.coroutines.CancellationException) {
            if (!stopGenerating) {
                throw e
            }
            Log.d(TAG, "Agent generation cancelled", e)
            HashMap<String, Any>().apply {
                put("error", true)
                put("message", "已停止")
                put("response", "已停止")
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
        chatDataManager!!.addOrUpdateSession(
            sessionId,
            modelId,
            if (agentEnabled) {
                com.alibaba.mnnllm.android.chat.model.ChatDatabaseHelper.SESSION_MODE_AGENT
            } else {
                com.alibaba.mnnllm.android.chat.model.ChatDatabaseHelper.SESSION_MODE_NORMAL
            }
        )
        chatDataManager!!.updateSessionName(this.sessionId!!, this.sessionName)
    }


    suspend fun requestGenerate(userData: ChatDataItem, generateListener: GenerateListener): HashMap<String, Any> {
        this.generateListener = generateListener
        val prompt = PromptUtils.generateUserPrompt(userData)
        
        // Ensure user input is saved first
        try {
            if (this.sessionName.isNullOrEmpty()) {
                this.sessionName = SessionUtils.generateSessionName(userData)
                updateSession(sessionId!!, chatActivity.modelId ?: modelId, sessionName!!)
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

        override fun onAgentStatus(status: String) {
            chatActivity.lifecycleScope.launch {
                chatActivity.onAgentStatus(status)
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
                    if (chatSession is com.alibaba.mnnllm.android.llm.LlmSession) {
                        if (!ServerEventManager.getInstance().isServerRunning()) {
                            Log.d(TAG, "Final cleanup: Resetting and releasing LlmSession via LlmRuntimeController")
                            chatSession.reset()
                            ServiceLocator.getLlmRuntimeController().releaseSession()
                        } else {
                            Log.d(TAG, "LlmSession kept alive (API running)")
                        }
                    } else {
                        Log.d(TAG, "Final cleanup: Resetting and releasing non-LLM session")
                        chatSession.reset()
                        chatSession.release()
                    }
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
                val newSession = createNewModelSession(newModelItem, currentChatHistory)
                updateDatabaseForModelSwitch(oldSessionId, newModelItem.modelId!!)
                chatActivity.lifecycleScope.launch {
                    onSessionCreated(newSession)
                }
                if (newSession !is com.alibaba.mnnllm.android.llm.LlmSession ||
                    !(newSession as com.alibaba.mnnllm.android.llm.LlmSession).isModelLoaded()) {
                    newSession.load()
                }
                applySystemPromptForCurrentMode()
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
            if (chatSession is com.alibaba.mnnllm.android.llm.LlmSession) {
                ServiceLocator.getLlmRuntimeController().releaseSession()
            } else {
                chatSession.release()
            }
        }
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
            deferLoad = true
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
        fun onAgentStatus(status: String) {}
    }
}
