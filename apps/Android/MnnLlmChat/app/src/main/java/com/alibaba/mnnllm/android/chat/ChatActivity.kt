// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.chat

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.Menu
import android.view.MenuItem
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.coroutines.flow.filterIsInstance
import kotlinx.coroutines.flow.first
import com.alibaba.mls.api.ApplicationProvider
import com.alibaba.mnnllm.android.llm.ChatSession
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.BuildConfig
import com.alibaba.mnnllm.android.audio.AudioChunksPlayer
import com.alibaba.mnnllm.android.benchmark.BenchmarkModule
import com.alibaba.mnnllm.android.modelist.ModelListManager
import com.alibaba.mnnllm.android.utils.WavFileWriter
import com.alibaba.mnnllm.android.utils.FileUtils
import com.alibaba.mnnllm.android.chat.chatlist.ChatListComponent
import com.alibaba.mnnllm.android.chat.chatlist.ChatViewHolders
import com.alibaba.mnnllm.android.chat.input.ChatInputComponent
import com.alibaba.mnnllm.android.chat.model.ChatDataItem
import com.alibaba.mnnllm.android.databinding.ActivityChatBinding
import com.alibaba.mnnllm.android.llm.AudioDataListener
import com.alibaba.mnnllm.android.llm.LlmSession
import com.alibaba.mnnllm.android.mainsettings.MainSettings.isApiServiceEnabled
import com.alibaba.mnnllm.android.modelsettings.SettingsBottomSheetFragment
import com.alibaba.mnnllm.android.modelsettings.DiffusionSettingsBottomSheetFragment
import com.alibaba.mnnllm.api.openai.ui.ApiSettingsBottomSheetFragment
import com.alibaba.mnnllm.api.openai.ui.ApiConsoleBottomSheetFragment
import com.alibaba.mnnllm.android.utils.AudioPlayService
import com.alibaba.mnnllm.android.model.ModelTypeUtils
import com.alibaba.mnnllm.android.model.ModelUtils
import com.alibaba.mnnllm.android.utils.PreferenceUtils
import com.alibaba.mnnllm.api.openai.manager.ApiServiceManager
import com.alibaba.mnnllm.android.chat.voice.VoiceChatFragment
import com.alibaba.mnnllm.android.chat.voice.VoiceModelsChecker
import com.alibaba.mnnllm.android.chat.voice.VoiceModelMarketBottomSheet
import com.alibaba.mnnllm.android.modelist.ModelItemWrapper
import com.alibaba.mnnllm.android.utils.CrashReportContext
import com.alibaba.mnnllm.android.utils.ConfigInfoDialog
import kotlinx.coroutines.MainScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.filter
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import java.io.File
import java.text.DateFormat
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * @description:
 * lifecycle:
 * setupView
 * setupSession
 * sessionLoaded
 */
class ChatActivity : AppCompatActivity() {
    var isGenerating: Boolean
        get() = _isGenerating.value
        set(value) {
            _isGenerating.value = value
        }
    var dateFormat: DateFormat? = null
    var sessionId: String? = null
        private set
    var isLoading = false
    var isAudioModel = false
    var isDiffusion = false
    var chatSession: ChatSession? = null

    private val _isGenerating = MutableStateFlow(false)
    private var layoutModelLoading: View? = null
    var modelName: String = ""
    var modelId: String? = null
    private var currentUserMessage: ChatDataItem? = null
    private var sessionName: String? = null
    private var useDiffusionWaitHint: Boolean = false
    private var hasMeaningfulDiffusionProgress: Boolean = false
    private var showDiffusionPercentWhenZero: Boolean = false
    private lateinit var binding: ActivityChatBinding
    private var audioPlayer: AudioChunksPlayer? = null
    private lateinit var chatPresenter: ChatPresenter
    private var chatInputModule: ChatInputComponent? = null
    lateinit var chatListComponent: ChatListComponent

    // Real-time audio playback settings
    private var isRealTimePlayback = true
    private var wavFileWriter: WavFileWriter? = null
    private var bufferedAudioFilePath: String? = null

    private var benchmarkModule: BenchmarkModule = BenchmarkModule(activity = this)
    private lateinit var voiceModelsChecker: VoiceModelsChecker
    private var isMockStreamSession: Boolean = false
    private val mockStreamHandler = Handler(Looper.getMainLooper())
    private var mockStreamRunnable: Runnable? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityChatBinding.inflate(layoutInflater)
        setContentView(binding.root)
        val toolbar = binding.toolbar
        setSupportActionBar(toolbar)

        this.modelName = intent.getStringExtra("modelName")?:""
        this.modelId = intent.getStringExtra("modelId")
        if (this.modelName.isEmpty() || this.modelId.isNullOrEmpty()) {
            finish()
            return
        }
        CrashReportContext.setCurrentModel(this.modelId, intent.getStringExtra("chatSessionId"))
        dateFormat = SimpleDateFormat("hh:mm aa", Locale.getDefault())
        layoutModelLoading = findViewById(R.id.layout_model_loading)
        updateActionBar()
        binding.modelSwitcher.setOnClickListener {
            if (!isDiffusion) {
                showModelSelectionDialog()
            }
        }
        setupView(this.modelId!!, this.modelName)
        if (shouldStartMockStream()) {
            isMockStreamSession = true
            chatListComponent.setup(modelName, emptyList())
            startMockStream()
            return
        }
        this.setupSession()
        initializeVoiceModelsChecker()
    }

    private fun setupView(modelId:String, modelName: String) {
        this.modelId = modelId
        this.modelName = modelName
        isDiffusion = ModelTypeUtils.isDiffusionModel(modelName)
        isAudioModel = ModelTypeUtils.isAudioModel(modelId)
        binding.modelSwitcher.text = modelName
        
        // Hide model switcher click functionality for diffusion models
        val dropdownArrow = binding.modelSwitcher.findViewById<View>(R.id.iv_dropdown_arrow)
        if (isDiffusion) {
            binding.modelSwitcher.isClickable = false
            binding.modelSwitcher.isFocusable = false
            binding.modelSwitcher.background = null
            dropdownArrow?.visibility = View.GONE
        } else {
            binding.modelSwitcher.isClickable = true
            binding.modelSwitcher.isFocusable = true
//            binding.modelSwitcher.setBackgroundResource(R.drawable.bg_rounded_dropdown)
            dropdownArrow?.visibility = View.VISIBLE
        }
        
        chatPresenter = ChatPresenter(this, modelName, modelId)
        setChatPresenter(chatPresenter)
        chatInputModule = ChatInputComponent(this, binding, modelId, modelName)
        setupChatListComponent()
        setupInputModule()
        binding.modelSwitcher.text = modelName
    }

    private fun onSessionCreated() {
        val history = chatSession!!.getHistory()
        Log.d(TAG, "onSessionCreated: setting up UI with ${history?.size ?: 0} history items, isDiffusion=$isDiffusion")
        chatListComponent.setup(modelName, history)
    }

    private fun setupChatListComponent() {
        chatListComponent = ChatListComponent(this, this.dateFormat!!, binding)
    }

    private fun updateActionBar() {
        if (supportActionBar != null) {
            supportActionBar!!.setDisplayHomeAsUpEnabled(true)
            supportActionBar!!.setDisplayShowTitleEnabled(false)
        }
    }

    private fun setupInputModule() {
        this.chatInputModule!!.apply {
            setOnThinkingModeChanged {isThinking ->
                Log.d(TAG, "isThinking: $isThinking")
                (chatSession as LlmSession).updateThinking(isThinking)
            }
            setOnAudioOutputModeChanged {
                chatPresenter.setEnableAudioOutput(it)
            }
            setOnSendMessage{
                lifecycleScope.launch {
                    this@ChatActivity.handleSendMessage(it)
                }
            }
            setOnStopGenerating{
                chatPresenter.stopGenerate()
            }
        }
    }

    private fun setupSession() {
        chatSession = chatPresenter.createSession()
        sessionId = chatSession!!.sessionId
        CrashReportContext.setCurrentModel(modelId, sessionId)
        onSessionCreated()
        Log.d(TAG, "current SessionId: $sessionId")
        chatPresenter.load()
    }

    private fun shouldStartMockStream(): Boolean {
        if (!BuildConfig.DEBUG) {
            return false
        }
        return intent.getBooleanExtra(EXTRA_MOCK_STREAM_ENABLE, false)
    }

    private fun startMockStream() {
        val textLength = intent.getIntExtra(
            EXTRA_MOCK_STREAM_TEXT_LENGTH,
            DEFAULT_MOCK_STREAM_TEXT_LENGTH
        ).coerceAtLeast(1)
        val intervalMs = intent.getLongExtra(
            EXTRA_MOCK_STREAM_INTERVAL_MS,
            DEFAULT_MOCK_STREAM_INTERVAL_MS
        ).coerceAtLeast(1L)
        val lineWidth = intent.getIntExtra(
            EXTRA_MOCK_STREAM_LINE_WIDTH,
            DEFAULT_MOCK_STREAM_LINE_WIDTH
        ).coerceAtLeast(1)
        val detachAtChars = intent.getIntExtra(
            EXTRA_MOCK_STREAM_DETACH_AT_CHARS,
            DEFAULT_MOCK_STREAM_DETACH_AT_CHARS
        )
        val pauseOnDetach = intent.getBooleanExtra(
            EXTRA_MOCK_STREAM_PAUSE_ON_DETACH,
            false
        )
        val mockText = resolveMockStreamText(textLength, lineWidth)
        val userData = createUserMessage("mock long stream")
        onGenerateStart(userData)
        val processor = GenerateResultProcessor()
        processor.generateBegin()
        mockStreamRunnable?.let(mockStreamHandler::removeCallbacks)

        var nextIndex = 0
        var didDetachFromBottom = false
        var mockStreamPaused = false
        chatListComponent.setOnResumeAutoScrollListener {
            mockStreamPaused = false
        }
        val streamRunnable = object : Runnable {
            override fun run() {
                if (isDestroyed) {
                    return
                }
                if (mockStreamPaused) {
                    mockStreamHandler.postDelayed(this, intervalMs)
                    return
                }
                if (nextIndex >= mockText.length) {
                    val bench = HashMap<String, Any>().apply {
                        put("response", processor.getRawResult())
                        put("prompt_len", 1L)
                        put("decode_len", mockText.length.toLong())
                        put("prefill_time", 1L)
                        put("decode_time", mockText.length.toLong() * intervalMs * 1000L)
                    }
                    onGenerateFinished(bench)
                    mockStreamRunnable = null
                    return
                }
                val chunk = mockText[nextIndex].toString()
                nextIndex += 1
                processor.process(chunk)
                onLlmGenerateProgress(chunk, processor)
                if (!didDetachFromBottom && detachAtChars > 0 && nextIndex >= detachAtChars) {
                    didDetachFromBottom = true
                    if (pauseOnDetach) {
                        mockStreamPaused = true
                    }
                    chatListComponent.detachFromBottomForTest()
                }
                mockStreamHandler.postDelayed(this, intervalMs)
            }
        }
        mockStreamRunnable = streamRunnable
        mockStreamHandler.post(streamRunnable)
    }

    private fun resolveMockStreamText(textLength: Int, lineWidth: Int): String {
        val inlineContent = intent.getStringExtra(EXTRA_MOCK_STREAM_CONTENT)
        if (!inlineContent.isNullOrEmpty()) {
            return inlineContent
        }

        val contentFile = intent.getStringExtra(EXTRA_MOCK_STREAM_CONTENT_FILE)
        if (!contentFile.isNullOrEmpty()) {
            return try {
                File(contentFile).readText()
            } catch (e: Exception) {
                Log.e(TAG, "Failed to read mock stream content file: $contentFile", e)
                buildMockStreamText(textLength, lineWidth)
            }
        }

        return buildMockStreamText(textLength, lineWidth)
    }

    private fun buildMockStreamText(length: Int, lineWidth: Int): String {
        val base = "mockstreamoutput"
        val builder = StringBuilder(length + 64)
        var sourceIndex = 0
        var currentLineWidth = 0
        while (builder.length < length) {
            if (currentLineWidth >= lineWidth) {
                builder.append("\n\n")
                currentLineWidth = 0
                if (builder.length >= length) {
                    break
                }
            }
            val nextChar = base[sourceIndex]
            builder.append(nextChar)
            sourceIndex = (sourceIndex + 1) % base.length
            currentLineWidth = if (nextChar == '\n') 0 else currentLineWidth + 1
        }
        if (builder.length > length) {
            builder.setLength(length)
        }
        return builder.toString()
    }

    private fun setupOmni() {
        audioPlayer = AudioChunksPlayer()
        audioPlayer!!.sampleRate = 24000  // Use same sample rate as original AudioPlayer
        audioPlayer!!.start()
        
        (chatSession as LlmSession).setAudioDataListener(object : AudioDataListener {
            override fun onAudioData(data: FloatArray, isEnd: Boolean): Boolean {
                this@ChatActivity.lifecycleScope.launch {
                    if (isRealTimePlayback) {
                        // Real-time playback mode: play immediately but also save for replay
                        audioPlayer?.playChunk(data)
                        
                        // Also save audio data for potential replay
                        saveAudioDataForReplay(data, isEnd)
                        
                        if (isEnd) {
                            audioPlayer?.endChunk()
                        }
                    } else {
                        // Buffered playback mode: collect all chunks first
                        handleBufferedAudioData(data, isEnd)
                    }
                }
                return chatPresenter.stopGenerating
            }
        })
    }

    private suspend fun handleBufferedAudioData(data: FloatArray, isEnd: Boolean) {
        // Initialize WAV writer if this is the first chunk
        if (wavFileWriter == null) {
            bufferedAudioFilePath = FileUtils.generateDestAudioFilePath(this, sessionId!!)
            wavFileWriter = WavFileWriter(bufferedAudioFilePath!!, 24000, 1, 16)
            Log.d(TAG, "Initialized WAV writer for buffered playback: $bufferedAudioFilePath")
        }

        // Add audio chunk to WAV writer
        wavFileWriter?.addAudioChunk(data)

        if (isEnd) {
            // Write all chunks to WAV file and play
            val success = wavFileWriter?.writeToFile() ?: false
            if (success && bufferedAudioFilePath != null) {
                Log.d(TAG, "WAV file written successfully, starting playback")
                
                // Update the current chat item with audio info
                val currentItem = chatListComponent.recentItem
                currentItem?.let { item ->
                    item.hasOmniAudio = true
                    item.audioUri = Uri.fromFile(java.io.File(bufferedAudioFilePath!!))
                    chatListComponent.updateAssistantResponse(item)
                }
                
                // Auto-play the audio file
                playWavFile(bufferedAudioFilePath!!)
            } else {
                Log.e(TAG, "Failed to write WAV file")
            }
            
            // Clean up
            wavFileWriter?.clear()
            wavFileWriter = null
            bufferedAudioFilePath = null
        }
    }

    private suspend fun saveAudioDataForReplay(data: FloatArray, isEnd: Boolean) {
        // Initialize WAV writer for replay if this is the first chunk
        if (wavFileWriter == null) {
            bufferedAudioFilePath = FileUtils.generateDestAudioFilePath(this, sessionId!!)
            wavFileWriter = WavFileWriter(bufferedAudioFilePath!!, 24000, 1, 16)
            Log.d(TAG, "Initialized WAV writer for audio replay: $bufferedAudioFilePath")
        }

        // Add audio chunk to WAV writer
        wavFileWriter?.addAudioChunk(data)

        if (isEnd) {
            // Write all chunks to WAV file for later replay
            val success = wavFileWriter?.writeToFile() ?: false
            if (success && bufferedAudioFilePath != null) {
                Log.d(TAG, "Audio saved for replay: $bufferedAudioFilePath")
                
                // Update the current chat item with audio info  
                val currentItem = chatListComponent.recentItem
                currentItem?.let { item ->
                    item.hasOmniAudio = true
                    item.audioUri = Uri.fromFile(java.io.File(bufferedAudioFilePath!!))
                    chatListComponent.updateAssistantResponse(item)
                }
            } else {
                Log.e(TAG, "Failed to save audio for replay")
            }
            
            // Clean up
            wavFileWriter?.clear()
            wavFileWriter = null
            bufferedAudioFilePath = null
        }
    }

    private suspend fun playWavFile(filePath: String) {
        try {
            Log.d(TAG, "playWavFile : $filePath")
            // Reset audio player for file playback
            audioPlayer?.reset()
            withContext(Dispatchers.Main) {
                val audioPlayService = AudioPlayService.instance
                audioPlayService?.playAudio(filePath, object : AudioPlayService.AudioPlayerCallback {
                    override fun onPlayStart() {
                        Log.d(TAG, "Started playing buffered audio file")
                    }

                    override fun onPlayFinish() {
                        Log.d(TAG, "Completed playing buffered audio file")
                    }

                    override fun onPlayError() {
                        Log.e(TAG, "Error playing buffered audio file")
                    }

                    override fun onPlayProgress(progress: Float) {
                        // Not needed for our use case
                    }
                })
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error playing WAV file: $filePath", e)
        }
    }

    fun onLoadingChanged(loading: Boolean) {
        isLoading = loading
        this.chatInputModule!!.onLoadingStatesChanged(loading)
        layoutModelLoading!!.visibility =
            if (loading) View.VISIBLE else View.GONE
        if (supportActionBar != null) {
            supportActionBar!!.setDisplayHomeAsUpEnabled(true)
            binding.modelSwitcher.text = modelName
        }
        if (!loading) {
            if (chatSession!!.supportOmni) {
                setupOmni()
            }
            // Check API service settings and start service
            if (isApiServiceEnabled(this)) {
                ApiServiceManager.startApiService(this, modelId)
            }
        }
    }

    /**
     * Called when model load fails (e.g. native init returned null). Shows error and finishes.
     */
    fun onModelLoadFailed(errorMessage: String) {
        Log.e(TAG, "Model load failed: $errorMessage")
        Toast.makeText(this, getString(R.string.model_load_failed, errorMessage), Toast.LENGTH_LONG).show()
        finish()
    }

    override fun onCreateOptionsMenu(menu: Menu): Boolean {
        menuInflater.inflate(R.menu.menu_chat, menu)
        menu.findItem(R.id.show_performance_metrics)
            .setChecked(
                PreferenceUtils.getBoolean(
                    this,
                    PreferenceUtils.KEY_SHOW_PERFORMACE_METRICS,
                    true
                )
            )
        menu.findItem(R.id.menu_item_model_settings).isVisible = true
        menu.findItem(R.id.menu_item_benchmark_test).isVisible = benchmarkModule.enabled
        // Voice chat is only available for non-diffusion models
        menu.findItem(R.id.start_voice_chat).isVisible = !isDiffusion
        // Real-time audio playback is only available for Omni models
        val isOmniModel = ModelTypeUtils.isOmni(modelName)
        menu.findItem(R.id.realtime_audio_playback).isVisible = false
        menu.findItem(R.id.realtime_audio_playback).isChecked = false
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        if (item.itemId == R.id.start_new_chat) {
            handleNewSession()
        } else if (item.itemId == R.id.start_voice_chat) {
            handleVoiceChatClick()
        } else if (item.itemId == R.id.show_performance_metrics) {
            item.setChecked(!item.isChecked)
            chatListComponent.toggleShowPerformanceMetrics(item.isChecked)
        } else if (item.itemId == R.id.realtime_audio_playback) {
            item.setChecked(!item.isChecked)
            setRealTimeAudioPlayback(item.isChecked)
        } else if (item.itemId == android.R.id.home) {
            finish()
        } else if (item.itemId == R.id.menu_item_model_settings) {
            val session = chatSession
            if (session is LlmSession) {
                SettingsBottomSheetFragment().apply {
                    setModelId(modelId!!)
                    setConfigPath(intent.getStringExtra("configFilePath"))
                    setSession(session)
                    addOnSettingsDoneListener{needRecreate->
                        if (needRecreate) {
                            recreate()
                        }
                    }
                }.show(supportFragmentManager, SettingsBottomSheetFragment.TAG)
            } else {
                // For Sana and other diffusion models
                DiffusionSettingsBottomSheetFragment().apply {
                    setModelId(modelId!!)
                    setConfigPath(intent.getStringExtra("configFilePath"))
                    addOnSettingsDoneListener{needRecreate->
                        if (needRecreate) {
                            recreate()
                        }
                    }
                }.show(supportFragmentManager, DiffusionSettingsBottomSheetFragment.TAG)
            }
            return true
        } else if (item.itemId == R.id.menu_item_benchmark_test) {
            chatSession!!.setKeepHistory(false)
            benchmarkModule.start(waitForLastCompleted = {
                waitForGeneratingFinished()
            }, handleSendMessage = { message ->
                chatSession!!.reset()
                return@start handleSendMessage(createUserMessage(message))
            })
        } else if (item.itemId == R.id.menu_item_api_settings) {
            ApiSettingsBottomSheetFragment().show(supportFragmentManager, "ApiSettingsBottomSheetFragment")
            return true
        } else if (item.itemId == R.id.menu_item_api_console) {
            ApiConsoleBottomSheetFragment.newInstance(this).show(supportFragmentManager, "ApiConsoleBottomSheetFragment")
            return true
        } else if (item.itemId == R.id.menu_item_config_info) {
            showConfigInfo()
            return true
        }
        return super.onOptionsItemSelected(item)
    }

    private fun createUserMessage(text:String):ChatDataItem {
        val userMessage = ChatDataItem(ChatViewHolders.USER)
        userMessage.text = text
        userMessage.time = dateFormat!!.format(Date())
        return userMessage
    }

    private suspend fun waitForGeneratingFinished() {
        if (_isGenerating.value) {
            _isGenerating
                .filter { !it }
                .first()
        }
    }



    private fun handleNewSession() {
        if (!isGenerating) {
            currentUserMessage = null
            if (chatListComponent.reset()) {
                Toast.makeText(this, R.string.new_conversation_started, Toast.LENGTH_LONG).show()
            }
            this.sessionName = null
            chatPresenter.reset{newSessionId ->
                sessionId = newSessionId
                CrashReportContext.setCurrentModel(modelId, sessionId)
            }
        } else {
            Toast.makeText(this, "Cannot Create New Session when generating", Toast.LENGTH_LONG).show()
        }
    }

    private fun setIsGenerating(isGenerating: Boolean) {
        this.isGenerating = isGenerating
        this.chatInputModule!!.setIsGenerating(isGenerating)
    }

    @Deprecated("This method has been deprecated in favor of using the Activity Result API\n      which brings increased type safety via an {@link ActivityResultContract} and the prebuilt\n      contracts for common intents available in\n      {@link androidx.activity.result.contract.ActivityResultContracts}, provides hooks for\n      testing, and allow receiving results in separate, testable classes independent from your\n      activity. Use\n      {@link #registerForActivityResult(ActivityResultContract, ActivityResultCallback)}\n      with the appropriate {@link ActivityResultContract} and handling the result in the\n      {@link ActivityResultCallback#onActivityResult(Object) callback}.")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        this.chatInputModule!!.handleResult(requestCode, resultCode, data)
    }
    
    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        // Forward permission results to the attachment picker module
        this.chatInputModule?.let { inputModule ->
            if (inputModule is ChatInputComponent) {
                inputModule.attachmentPickerModule?.onRequestPermissionsResult(requestCode, permissions, grantResults)
            }
        }
    }

    private suspend fun handleSendMessage(userData: ChatDataItem): HashMap<String, Any> {
        return chatPresenter.sendMessage(userData)
    }

    override fun onDestroy() {
        super.onDestroy()
        mockStreamRunnable?.let(mockStreamHandler::removeCallbacks)
        mockStreamRunnable = null
        audioPlayer?.destroy()
        audioPlayer = null
        
        // Clean up WAV writer resources
        wavFileWriter?.clear()
        wavFileWriter = null
        bufferedAudioFilePath = null
        
        if (::chatPresenter.isInitialized) {
            chatPresenter.destroy()
        }
        MainScope().launch {
            ApiServiceManager.stopApiService(ApplicationProvider.get())
        }
    }

    override fun onStop() {
        super.onStop()
        AudioPlayService.instance!!.destroy()
    }
    fun onGenerateStart(userData: ChatDataItem) {
        chatListComponent.onStartSendMessage(userData)
        setIsGenerating(true)
        val recentItem = chatListComponent.recentItem
        recentItem?.loading = true
        recentItem?.forceShowLoadingWithText = false
        useDiffusionWaitHint = DiffusionWaitHintPolicy.shouldShowWaitHint(modelName)
        hasMeaningfulDiffusionProgress = false
        showDiffusionPercentWhenZero = ModelTypeUtils.isSanaModel(modelName)
        if (useDiffusionWaitHint && recentItem != null) {
            recentItem.text = getString(R.string.diffusion_wait_hint_no_progress)
            recentItem.displayText = recentItem.text
            recentItem.forceShowLoadingWithText = true
            chatListComponent.updateAssistantResponse(recentItem)
        }
    }

    /**
     * Set real-time audio playback mode
     * @param realTime true for real-time playback, false for buffered playback
     */
    fun setRealTimeAudioPlayback(realTime: Boolean) {
        Log.d(TAG, "Setting real-time audio playback: $realTime")
        isRealTimePlayback = realTime
        
        // Clean up existing WAV writer if switching modes
        if (!realTime) {
            wavFileWriter?.clear()
            wavFileWriter = null
            bufferedAudioFilePath = null
        }
    }

    /**
     * Get current real-time audio playback mode
     */
    fun isRealTimeAudioPlayback(): Boolean = isRealTimePlayback

    fun onLlmGenerateProgress(progress: String?, generateResultProcessor:GenerateResultProcessor) {
        val chatDataItem = chatListComponent.recentItem!!
        chatDataItem.thinkingText = generateResultProcessor.getThinkingContent()
        chatDataItem.displayText = generateResultProcessor.getNormalOutput()
        chatDataItem.text = generateResultProcessor.getRawResult()
        chatDataItem.thinkingFinishedTime = if (generateResultProcessor.thinkTime > 0) generateResultProcessor.thinkTime else -1
        chatListComponent.updateAssistantResponse(chatDataItem)
    }

    fun onDiffusionGenerateProgress(progress: String?, diffusionDestPath: String?) {
        val chatDataItem = chatListComponent.recentItem
        if (chatDataItem == null) {
            Log.e(TAG, "onDiffusionGenerateProgress: recentItem is null")
            return
        }
        
        try {
            // Ensure the item has a proper timestamp if not set
            if (chatDataItem.time.isNullOrEmpty()) {
                chatDataItem.time = dateFormat?.format(java.util.Date())
            }
            
            if ("100" == progress) {
                chatDataItem.text = getString(R.string.diffusion_generated_message)
                chatDataItem.displayText = chatDataItem.text
                chatDataItem.forceShowLoadingWithText = false
                if (!diffusionDestPath.isNullOrEmpty()) {
                    chatDataItem.imageUri = Uri.fromFile(java.io.File(diffusionDestPath))
                    Log.d(TAG, "onDiffusionGenerateProgress: Set imageUri to ${chatDataItem.imageUri}")
                } else {
                    Log.w(TAG, "onDiffusionGenerateProgress: diffusionDestPath is null or empty")
                }
            } else {
                val numericProgress = progress?.toIntOrNull()
                if (DiffusionProgressHintPolicy.isMeaningfulProgress(progress)) {
                    hasMeaningfulDiffusionProgress = true
                }
                chatDataItem.text = if (useDiffusionWaitHint) {
                    val shouldShowPercent = hasMeaningfulDiffusionProgress || (showDiffusionPercentWhenZero && numericProgress != null)
                    if (shouldShowPercent) {
                        getString(R.string.diffusion_opencl_wait_hint, (numericProgress ?: 0).toString())
                    } else {
                        getString(R.string.diffusion_wait_hint_no_progress)
                    }
                } else {
                    getString(R.string.diffusion_generate_progress, progress ?: "0")
                }
                chatDataItem.displayText = chatDataItem.text
                chatDataItem.forceShowLoadingWithText = useDiffusionWaitHint
            }
            chatListComponent.updateAssistantResponse(chatDataItem)
        } catch (e: Exception) {
            Log.e(TAG, "onDiffusionGenerateProgress: Error updating progress", e)
        }
    }

    fun onGenerateFinished(benchMarkResult: HashMap<String, Any>) {
        setIsGenerating(false)
        val recentItem = chatListComponent.recentItem!!
        recentItem.loading = false
        recentItem.forceShowLoadingWithText = false
        useDiffusionWaitHint = false
        hasMeaningfulDiffusionProgress = false
        showDiffusionPercentWhenZero = false
        
        // Handle error cases
        if (benchMarkResult.containsKey("error") && benchMarkResult["error"] as Boolean) {
            val errorMessage = benchMarkResult["message"] as? String ?: "generation_failed"
            recentItem.text = errorMessage
            recentItem.displayText = errorMessage
            Log.e(TAG, "Generation failed: $errorMessage")
        } else {
            // Normal success case - set response if available
            val response = benchMarkResult["response"] as? String
            if (!response.isNullOrEmpty() && recentItem.text.isNullOrEmpty()) {
                recentItem.text = response
                recentItem.displayText = response
            }
        }
        
        recentItem.benchmarkInfo = ModelUtils.generateBenchMarkString(benchMarkResult)
        chatListComponent.updateAssistantResponse(recentItem)

        if (isMockStreamSession) {
            return
        }
        
        // Always save to database, even for errors, to maintain conversation history
        try {
            chatPresenter.saveResponseToDatabase(recentItem)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save response to database", e)
        }
    }

    /**
     * Handle generation stop request from voice chat or other components
     * This method triggers the same stop logic as the UI stop button
     */
    fun onStopGenerationRequested() {
        Log.d(TAG, "Stop generation requested from external component")
        if (isGenerating) {
            // Trigger the same stop logic as the UI stop button
            chatPresenter.stopGenerate()
            
            // Update UI state immediately
            setIsGenerating(false)
            val recentItem = chatListComponent.recentItem
            recentItem?.loading = false
            recentItem?.forceShowLoadingWithText = false
            useDiffusionWaitHint = false
            hasMeaningfulDiffusionProgress = false
            showDiffusionPercentWhenZero = false
            
            Log.d(TAG, "Generation stopped by external request")
        } else {
            Log.d(TAG, "No active generation to stop")
        }
    }

    val sessionDebugInfo: String
        get() = chatSession!!.debugInfo

    private fun showConfigInfo() {
        val session = chatSession
        if (session is LlmSession) {
            val configJson = session.dumpConfig()
            ConfigInfoDialog.show(this, configJson)
        } else {
            Toast.makeText(this, "Config info not available for this model type", Toast.LENGTH_SHORT).show()
        }
    }

    private fun initializeVoiceModelsChecker() {
        Log.d(TAG, "Initializing VoiceModelsChecker")
        try {
            voiceModelsChecker = VoiceModelsChecker(this)
            Log.d(TAG, "VoiceModelsChecker initialized successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize VoiceModelsChecker", e)
        }
    }

    private fun handleVoiceChatClick() {
        Log.d(TAG, "handleVoiceChatClick called")
        
        if (isGenerating) {
            Toast.makeText(this, "Cannot start voice chat when generating", Toast.LENGTH_LONG).show()
            return
        }
        
        try {
            // Check if voice chat models are ready
            val isReady = voiceModelsChecker.isVoiceChatReady()
            Log.d(TAG, "Voice models ready: $isReady")
            
            if (isReady) {
                // All voice models are ready, start voice chat
                Log.d(TAG, "Starting voice chat")
                startVoiceChat()
            } else {
                // Models are not ready - show the voice model market to let user manage downloads
                Log.d(TAG, "Models not ready, showing voice model market")
                showVoiceModelMarket()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error in handleVoiceChatClick", e)
            Toast.makeText(this, "Error: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }
    
    private fun showVoiceModelMarket() {
        Log.d(TAG, "showVoiceModelMarket called")
        try {
            val message = getString(R.string.voice_chat_setup_message)
            val bottomSheet = VoiceModelMarketBottomSheet.newInstance(message)
            Log.d(TAG, "Created VoiceModelMarketBottomSheet: $bottomSheet")
            
            // Set callback to check if models are ready after bottom sheet is dismissed
            bottomSheet.setOnDismissCallback {
                Log.d(TAG, "VoiceModelMarketBottomSheet dismissed, checking if models are ready")
                val isReady = voiceModelsChecker.isVoiceChatReady()
                Log.d(TAG, "Voice models ready after dismissal: $isReady")
                
                if (isReady) {
                    Log.d(TAG, "Starting voice chat after successful model setup")
                    startVoiceChat()
                } else {
                    Log.d(TAG, "Voice models still not ready after dismissal")
                    val status = voiceModelsChecker.getVoiceChatStatus()
                    Log.d(TAG, "Voice chat status: $status")
                }
            }
            
            Log.d(TAG, "SupportFragmentManager: $supportFragmentManager")
            bottomSheet.show(supportFragmentManager, VoiceModelMarketBottomSheet.TAG)
            Log.d(TAG, "BottomSheet show() called successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Error showing voice model market", e)
            Toast.makeText(this, "Failed to show voice model market: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }
    
    private fun updateVoiceChatButtonState() {
        // Update the voice chat button state in the options menu
        invalidateOptionsMenu()
    }
    


    fun startVoiceChat() {
        // Start a new session for voice chat, similar to handleNewSession
        if (!isGenerating) {
            currentUserMessage = null
            if (chatListComponent.reset()) {
                Toast.makeText(this, R.string.new_conversation_started, Toast.LENGTH_LONG).show()
            }
            this.sessionName = null
            chatPresenter.reset { newSessionId ->
                sessionId = newSessionId
                CrashReportContext.setCurrentModel(modelId, sessionId)
                // Create voice chat fragment with the new session
                val voiceChatFragment = VoiceChatFragment.newInstance(modelName, modelId!!, chatPresenter)
                supportFragmentManager.beginTransaction()
                    .replace(android.R.id.content, voiceChatFragment)
                    .addToBackStack(VoiceChatFragment.TAG)
                    .commit()
            }
        } else {
            Toast.makeText(this, "Cannot start voice chat when generating", Toast.LENGTH_LONG).show()
        }
    }

    
    private fun showModelSelectionDialog() {
        if (isGenerating) {
            Toast.makeText(this, "Please wait for current generation to complete before switching models", Toast.LENGTH_SHORT).show()
            return
        }
        lifecycleScope.launch {
            val availableModels = getAvailableModels()
            val modelFilter: (ModelItemWrapper) -> Boolean = { modelWrapper ->
                !ModelTypeUtils.isDiffusionModel(modelWrapper.displayName)
            }

            val selectModelFragment = SelectModelFragment.newInstance(availableModels, modelFilter, modelId)
            selectModelFragment.setOnModelSelectedListener { selectedModelWrapper ->
                handleModelSelection(selectedModelWrapper)
            }
            selectModelFragment.show(supportFragmentManager, SelectModelFragment.TAG)
        }
        
    }
    
    private fun getAvailableModels(): List<ModelItemWrapper> {
        // Get current models or wait for them
        return ModelListManager.getCurrentModels()?: emptyList()
    }
    
    private fun handleModelSelection(selectedModelWrapper: ModelItemWrapper) {
        val selectedModelItem = selectedModelWrapper.modelItem
        val selectedModelId = selectedModelItem.modelId
        val selectedModelName = selectedModelWrapper.displayName
        
        if (selectedModelId == modelId) {
            return
        }
        
        if (isGenerating) {
            Toast.makeText(this, "Please wait for current generation to complete before switching models", Toast.LENGTH_SHORT).show()
            return
        }
        val currentChatHistory = chatListComponent.getCurrentChatHistory()
        
        // Update model info without recreating components
        updateModelInfo(selectedModelId!!, selectedModelName)
        
        chatPresenter.switchModel(
            selectedModelItem,
            currentChatHistory,
            onSwitchComplete = { newSession ->
                Toast.makeText(this, "Model switched to $selectedModelName", Toast.LENGTH_SHORT).show()
            },
            onSwitchError = { error ->
                Log.e(TAG, "Error switching model", error)
                Toast.makeText(this, "Failed to switch model: ${error.message}", Toast.LENGTH_LONG).show()
            }, onSessionCreated = { newSession ->
                chatSession = newSession
                sessionId = newSession.sessionId
                CrashReportContext.setCurrentModel(modelId, sessionId)
                onSessionCreated()
            }
        )
    }
    
    /**
     * Update model information and refresh UI components without recreating them
     */
    private fun updateModelInfo(selectedModelId: String, selectedModelName: String) {
        this.modelId = selectedModelId
        this.modelName = selectedModelName
        CrashReportContext.setCurrentModel(this.modelId, sessionId)
        isDiffusion = ModelTypeUtils.isDiffusionModel(selectedModelName)
        isAudioModel = ModelTypeUtils.isAudioModel(selectedModelId)
        
        // Update model switcher text
        binding.modelSwitcher.text = selectedModelName
        
        // Update model switcher click functionality for diffusion models
        val dropdownArrow = binding.modelSwitcher.findViewById<View>(R.id.iv_dropdown_arrow)
        if (isDiffusion) {
            binding.modelSwitcher.isClickable = false
            binding.modelSwitcher.isFocusable = false
            binding.modelSwitcher.background = null
            dropdownArrow?.visibility = View.GONE
        } else {
            binding.modelSwitcher.isClickable = true
            binding.modelSwitcher.isFocusable = true
//            binding.modelSwitcher.setBackgroundResource(R.drawable.bg_rounded_dropdown)
            dropdownArrow?.visibility = View.VISIBLE
        }
        chatInputModule?.updateModel(selectedModelId, selectedModelName)
        chatListComponent.updateModel(selectedModelName)
    }

    companion object {
        const val TAG: String = "ChatActivity"
        const val EXTRA_MOCK_STREAM_ENABLE = "mock_stream_enable"
        const val EXTRA_MOCK_STREAM_CONTENT = "mock_stream_content"
        const val EXTRA_MOCK_STREAM_CONTENT_FILE = "mock_stream_content_file"
        const val EXTRA_MOCK_STREAM_TEXT_LENGTH = "mock_stream_text_length"
        const val EXTRA_MOCK_STREAM_INTERVAL_MS = "mock_stream_interval_ms"
        const val EXTRA_MOCK_STREAM_LINE_WIDTH = "mock_stream_line_width"
        const val EXTRA_MOCK_STREAM_DETACH_AT_CHARS = "mock_stream_detach_at_chars"
        const val EXTRA_MOCK_STREAM_PAUSE_ON_DETACH = "mock_stream_pause_on_detach"
        private const val DEFAULT_MOCK_STREAM_TEXT_LENGTH = 2000
        private const val DEFAULT_MOCK_STREAM_INTERVAL_MS = 30L
        private const val DEFAULT_MOCK_STREAM_LINE_WIDTH = 24
        private const val DEFAULT_MOCK_STREAM_DETACH_AT_CHARS = -1
        private var _chatPresenter: ChatPresenter? = null
        fun getChatPresenter(): ChatPresenter? {
            return this._chatPresenter
        }
        fun setChatPresenter(chatPresenter: ChatPresenter?) {
            _chatPresenter = chatPresenter
        }
    }
}
