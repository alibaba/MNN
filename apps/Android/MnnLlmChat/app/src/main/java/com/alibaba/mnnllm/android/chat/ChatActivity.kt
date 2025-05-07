// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.chat

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.Menu
import android.view.MenuItem
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.alibaba.mnnllm.android.ChatSession
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.audio.AudioPlayer
import com.alibaba.mnnllm.android.chat.chatlist.ChatListComponent
import com.alibaba.mnnllm.android.chat.input.ChatInputModule
import com.alibaba.mnnllm.android.chat.model.ChatDataItem
import com.alibaba.mnnllm.android.chat.model.ChatDataManager
import com.alibaba.mnnllm.android.databinding.ActivityChatBinding
import com.alibaba.mnnllm.android.modelsettings.SettingsBottomSheetFragment
import com.alibaba.mnnllm.android.utils.AudioPlayService
import com.alibaba.mnnllm.android.utils.ModelPreferences
import com.alibaba.mnnllm.android.utils.ModelUtils
import com.alibaba.mnnllm.android.utils.PreferenceUtils
import kotlinx.coroutines.MainScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.filter
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import java.text.DateFormat
import java.text.SimpleDateFormat
import java.util.Locale
import java.util.concurrent.Executors
import java.util.concurrent.ScheduledExecutorService

class ChatActivity : AppCompatActivity() {
    private lateinit var onStopGenerating: () -> Unit
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
    lateinit var chatSession: ChatSession

    private val _isGenerating = MutableStateFlow(false)
    private var layoutModelLoading: View? = null
    lateinit var modelName: String
    private var modelId: String? = null
    private var chatExecutor: ScheduledExecutorService? = null
    private var chatDataManager: ChatDataManager? = null
    private var currentUserMessage: ChatDataItem? = null
    private var sessionName: String? = null
    private val CONFIG_SHOW_CUSTOM_TOOLBAR = false
    private lateinit var binding: ActivityChatBinding
    private var audioPlayer: AudioPlayer? = null
    private lateinit var chatPresenter: ChatPresenter
    private lateinit var inputModule: ChatInputModule
    private lateinit var chatListComponent: ChatListComponent
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityChatBinding.inflate(layoutInflater)
        setContentView(binding.root)
        val toolbar = binding.toolbar
        setSupportActionBar(toolbar)

        modelName = intent.getStringExtra("modelName")?:""
        modelId = intent.getStringExtra("modelId")
        if (modelName.isEmpty() || modelId.isNullOrEmpty()) {
            finish()
        }
        chatPresenter = ChatPresenter(this, modelName, modelId!!)
        isDiffusion = ModelUtils.isDiffusionModel(modelName)
        isAudioModel = ModelUtils.isAudioModel(modelName)
        inputModule = ChatInputModule(this, binding, modelName,)
        layoutModelLoading = findViewById(R.id.layout_model_loading)
        updateActionBar()
        chatExecutor = Executors.newScheduledThreadPool(1)
        chatDataManager = ChatDataManager.getInstance(this)
        this.setupSession()
        dateFormat = SimpleDateFormat("hh:mm aa", Locale.getDefault())
        this.setupChatListComponent()
        setupInputModule()
    }

    private fun setupChatListComponent() {
        chatListComponent = ChatListComponent(this, binding)
    }

    private fun updateActionBar() {
        if (supportActionBar != null) {
            supportActionBar!!.setDisplayHomeAsUpEnabled(true)
            supportActionBar!!.setDisplayShowTitleEnabled(!CONFIG_SHOW_CUSTOM_TOOLBAR)
            supportActionBar!!.title = getString(R.string.app_name)
        }
    }

    private fun setupInputModule() {
        this.inputModule.apply {
            setOnThinkingModeChanged {isThinking ->
                chatSession.updateAssistantPrompt(if (isThinking) {
                    "<|im_start|>assistant\n%s<|im_end|>\n"
                } else {
                    "<|im_start|>assistant\n<think>\n</think>%s<|im_end|>\n"
                })
            }
            setOnRealSendMessage{
                this@ChatActivity.handleSendMessage(it)
            }
            this.setOnStopGenerating{
                chatPresenter.stopGenerate()
            }
        }
    }



    private fun setupSession() {
        chatSession = chatPresenter.createSession()
        sessionId = chatSession.sessionId
        Log.d(TAG, "current SessionId: $sessionId")
        chatPresenter.load()
    }

    private fun setupOmni() {
        audioPlayer = AudioPlayer()
        audioPlayer!!.start()
        chatSession.setAudioDataListener(object : ChatSession.AudioDataListener {
            override fun onAudioData(data: FloatArray, isEnd: Boolean): Boolean {
                MainScope().launch {
                    audioPlayer?.playChunk(data)
                }
                return true
            }
        })
    }

    fun setIsLoading(loading: Boolean) {
        isLoading = loading
        runOnUiThread {
            this.inputModule.onLoadingStatesChanged(loading)
            layoutModelLoading!!.visibility =
                if (loading) View.VISIBLE else View.GONE
            if (supportActionBar != null) {
                supportActionBar!!.setDisplayHomeAsUpEnabled(true)
                if (CONFIG_SHOW_CUSTOM_TOOLBAR) {
                } else {
                    supportActionBar!!.subtitle =
                    if (loading) getString(R.string.model_loading) else modelName
                }
            }
            if (!loading) {
                if (chatSession.supportOmni) {
                    setupOmni()
                }
            }
        }
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
        menu.findItem(R.id.menu_item_use_mmap).setChecked(
            ModelPreferences.getBoolean(
                this,
                modelId!!,
                ModelPreferences.KEY_USE_MMAP,
                false
            )
        )
        menu.findItem(R.id.menu_item_backend).setChecked(
            ModelPreferences.getBoolean(
                this,
                modelId!!,
                ModelPreferences.KEY_BACKEND,
                false
            )
        )

        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        if (item.itemId == R.id.start_new_chat) {
            handleNewSession()
        } else if (item.itemId == R.id.show_performance_metrics) {
            item.setChecked(!item.isChecked)
            chatListComponent.toggleShowPerformanceMetrics(item.isChecked)
        } else if (item.itemId == android.R.id.home) {
            finish()
        } else if (item.itemId == R.id.menu_item_clear_mmap_cache) {
            if (ModelPreferences.useMmap(this, modelId!!)) {
                Toast.makeText(this, R.string.mmap_cacche_cleared, Toast.LENGTH_LONG).show()
                chatSession.clearMmapCache()
                recreate()
            } else {
                Toast.makeText(this, R.string.mmap_not_used, Toast.LENGTH_SHORT).show()
            }
        } else if (item.itemId == R.id.menu_item_use_mmap) {
            item.setChecked(!item.isChecked)
            Toast.makeText(this, R.string.reloading_session, Toast.LENGTH_LONG).show()
            ModelPreferences.setBoolean(
                this,
                modelId!!,
                ModelPreferences.KEY_USE_MMAP,
                item.isChecked
            )
            recreate()
        } else if (item.itemId == R.id.menu_item_backend) {
            item.setChecked(!item.isChecked)
            Toast.makeText(this, R.string.reloading_session, Toast.LENGTH_LONG).show()
            ModelPreferences.setBoolean(this, modelId!!, ModelPreferences.KEY_BACKEND, item.isChecked)
            recreate()
        } else if (item.itemId == R.id.menu_item_model_settings) {
            val settingsSheet = SettingsBottomSheetFragment()
            settingsSheet.setSession(chatSession)
            settingsSheet.show(supportFragmentManager, SettingsBottomSheetFragment.TAG)
            return true
        }
        return super.onOptionsItemSelected(item)
    }


    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        this.inputModule.onRequestPermissionsResult(requestCode, permissions, grantResults)
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
            }
        } else {
            Toast.makeText(this, "Cannot Create New Session when generating", Toast.LENGTH_LONG).show()
        }
    }

    private suspend fun waitForGeneratingFinished() {
        if (_isGenerating.value) {
            _isGenerating
                .filter { !it }
                .first()
        }
    }

    private fun setIsGenerating(isGenerating: Boolean) {
        this.isGenerating = isGenerating
        this.inputModule.setIsGenerating(isGenerating)
    }

    @Deprecated("This method has been deprecated in favor of using the Activity Result API\n      which brings increased type safety via an {@link ActivityResultContract} and the prebuilt\n      contracts for common intents available in\n      {@link androidx.activity.result.contract.ActivityResultContracts}, provides hooks for\n      testing, and allow receiving results in separate, testable classes independent from your\n      activity. Use\n      {@link #registerForActivityResult(ActivityResultContract, ActivityResultCallback)}\n      with the appropriate {@link ActivityResultContract} and handling the result in the\n      {@link ActivityResultCallback#onActivityResult(Object) callback}.")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        this.inputModule.handleResult(requestCode, resultCode, data)
    }

    private fun handleSendMessage(userData: ChatDataItem) {
        setIsGenerating(true)
        chatListComponent.onStartSendMessage(userData)
        chatPresenter.onUserMessage(userData)

    }

    override fun onDestroy() {
        super.onDestroy()
        chatPresenter.destroy()
    }

    override fun onStop() {
        super.onStop()
        AudioPlayService.instance!!.destroy()
    }

    fun onGenerateStart() {
        setIsGenerating(true)
        val recentItem = chatListComponent.recentItem
        recentItem?.loading = true
    }

    fun onLlmGenerateProgress(progress: String?, generateResultProcessor:GenerateResultProcessor) {
        val chatDataItem = chatListComponent.recentItem!!
        chatDataItem.displayText = generateResultProcessor.getDisplayResult()
        chatDataItem.text = generateResultProcessor.getRawResult()
        chatListComponent.updateAssistantResponse(chatDataItem)
    }

    fun onGenerateProgress(progress: String?, diffusionDestPath: String?) {
        val chatDataItem = chatListComponent.recentItem!!
        if ("100" == progress) {
            chatDataItem.text = getString(R.string.diffusion_generated_message)
            chatDataItem.displayText = chatDataItem.text
            chatDataItem.imageUri = Uri.parse(diffusionDestPath)
        } else {
            chatDataItem.text = getString(R.string.diffusion_generate_progress, progress)
            chatDataItem.displayText = chatDataItem.text
        }
        chatListComponent.updateAssistantResponse(chatDataItem)
    }

    fun onGenerateFinished(benchMarkResult: HashMap<String, Any>) {
        setIsGenerating(false)
        val recentItem = chatListComponent.recentItem!!
        recentItem.loading = false
        recentItem.benchmarkInfo = ModelUtils.generateBenchMarkString(benchMarkResult)
        chatListComponent.updateAssistantResponse(recentItem)
    }

    val sessionDebugInfo: String
        get() = chatSession.debugInfo

    companion object {
        const val TAG: String = "ChatActivity"
    }
}