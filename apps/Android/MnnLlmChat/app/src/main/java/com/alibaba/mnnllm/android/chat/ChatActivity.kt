// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.chat

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.text.TextUtils
import android.util.Log
import android.view.Menu
import android.view.MenuItem
import android.view.View
import android.view.WindowInsets
import android.widget.EditText
import android.widget.ImageView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.alibaba.mnnllm.android.ChatService
import com.alibaba.mnnllm.android.ChatSession
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.audio.AudioPlayer
import com.alibaba.mnnllm.android.chat.GenerateResultProcessor.R1GenerateResultProcessor
import com.alibaba.mnnllm.android.databinding.ActivityChatBinding
import com.alibaba.mnnllm.android.modelsettings.SettingsBottomSheetFragment
import com.alibaba.mnnllm.android.utils.AudioPlayService
import com.alibaba.mnnllm.android.utils.FileUtils
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
import java.util.Date
import java.util.Locale
import java.util.Random
import java.util.concurrent.Executors
import java.util.concurrent.ScheduledExecutorService
import kotlin.math.abs

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
    var stopGenerating = false

    private val _isGenerating = MutableStateFlow(false)
    private lateinit var recyclerView: RecyclerView
    private var adapter: ChatRecyclerViewAdapter? = null
    private var layoutModelLoading: View? = null
    private lateinit var chatSession: ChatSession
    lateinit var modelName: String
    private var modelId: String? = null
    private var chatExecutor: ScheduledExecutorService? = null
    private var linearLayoutManager: LinearLayoutManager? = null
    private var chatDataManager: ChatDataManager? = null
    private var isUserScrolling = false
    private var attachmentPickerModule: AttachmentPickerModule? = null
    private var currentUserMessage: ChatDataItem? = null
    private var sessionName: String? = null
    private val CONFIG_SHOW_CUSTOM_TOOLBAR = false
    private lateinit var binding: ActivityChatBinding
    private var audioPlayer: AudioPlayer? = null
    private lateinit var chatPresenter: ChatPresenter
    private lateinit var inputModule: InputModule

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityChatBinding.inflate(layoutInflater)
        setContentView(binding.root)
        val toolbar = binding.toolbar
        setSupportActionBar(toolbar)

        chatPresenter = ChatPresenter(this)
        modelName = intent.getStringExtra("modelName")?:""
        if (modelName.isEmpty()) {
            finish()
        }
        modelId = intent.getStringExtra("modelId")
        isAudioModel = ModelUtils.isAudioModel(modelName)
        inputModule = InputModule(this, binding, modelName,)
        layoutModelLoading = findViewById(R.id.layout_model_loading)
        if (supportActionBar != null) {
            supportActionBar!!.setDisplayHomeAsUpEnabled(true)
            supportActionBar!!.setDisplayShowTitleEnabled(!CONFIG_SHOW_CUSTOM_TOOLBAR)
            supportActionBar!!.title = getString(R.string.app_name)
        }

        chatExecutor = Executors.newScheduledThreadPool(1)
        chatDataManager = ChatDataManager.getInstance(this)
        this.setupSession()
        dateFormat = SimpleDateFormat("hh:mm aa", Locale.getDefault())
        this.setupRecyclerView()
        smoothScrollToBottom()
        setupInputModule()
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
        }
    }

    fun regenerate() {
        stopGenerating = true
    }

    private fun setupSession() {
        val chatService = ChatService.provide()
        sessionId = intent.getStringExtra("chatSessionId")
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
        Log.d(TAG, "current SessionId: $sessionId")
        chatExecutor!!.submit {
            Log.d(TAG, "chatSession loading")
            setIsLoading(true)
            chatSession.load()
            if (chatSession.supportOmni) {
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
            setIsLoading(false)
            Log.d(TAG, "chatSession loaded")
        }
    }

    private fun setIsLoading(loading: Boolean) {
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
        }
    }

    private fun setupRecyclerView() {
        recyclerView = binding.recyclerView
        recyclerView.setItemAnimator(null)
        linearLayoutManager = LinearLayoutManager(this)
        recyclerView.setLayoutManager(linearLayoutManager)
        binding.layoutBottomContainer.addOnLayoutChangeListener { v, left, top, right, bottom, oldLeft, oldTop, oldRight, oldBottom ->
            val insets: WindowInsets? = v.rootWindowInsets
            val bottomInset = insets!!.systemWindowInsetBottom
            recyclerView.setPadding(recyclerView.paddingLeft, recyclerView.paddingTop, recyclerView.paddingRight,
                bottomInset +  binding.layoutBottomContainer.height)
            insets.consumeSystemWindowInsets()
        }
        adapter = ChatRecyclerViewAdapter(this, initData(), this.modelName)
        recyclerView.setAdapter(adapter)
        recyclerView.addOnScrollListener(object : RecyclerView.OnScrollListener() {
            override fun onScrollStateChanged(recyclerView: RecyclerView, newState: Int) {
                super.onScrollStateChanged(recyclerView, newState)
            }

            override fun onScrolled(recyclerView: RecyclerView, dx: Int, dy: Int) {
                super.onScrolled(recyclerView, dx, dy)
                if (abs(dy.toDouble()) > 0) {
                    isUserScrolling = true
                }
            }

            var isUserScrolling: Boolean = false
        })
    }

    private fun initData(): MutableList<ChatDataItem> {
        val data: MutableList<ChatDataItem> = ArrayList()
        data.add(ChatDataItem(dateFormat!!.format(Date()), ChatViewHolders.HEADER, ""))
        data.add(
            ChatDataItem(
                dateFormat!!.format(Date()), ChatViewHolders.ASSISTANT,
                getString(
                    if (ModelUtils.isDiffusionModel(modelName))
                        R.string.model_hello_prompt_diffusion else
                        R.string.model_hello_prompt,
                    modelName
                )
            )
        )
        val savedHistory = chatSession.savedHistory
        if (!savedHistory.isNullOrEmpty()) {
            data.addAll(savedHistory)
        }
        return data
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
            PreferenceUtils.setBoolean(
                this,
                PreferenceUtils.KEY_SHOW_PERFORMACE_METRICS,
                item.isChecked
            )
            adapter!!.notifyItemRangeChanged(0, adapter!!.itemCount)
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
            sessionId = chatSession.generateNewSession()
            this.sessionName = null
            chatExecutor!!.execute { chatSession.reset() }
            chatDataManager!!.deleteAllChatData(sessionId!!)
            if (adapter!!.reset()) {
                Toast.makeText(this, R.string.new_conversation_started, Toast.LENGTH_LONG).show()
            }
        } else {
            Toast.makeText(this, "Cannot Reset when generating", Toast.LENGTH_LONG).show()
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

    private fun smoothScrollToBottom() {
        Log.d(TAG, "smoothScrollToBottom")
        recyclerView.post {
            val position = adapter!!.itemCount - 1
            recyclerView.scrollToPosition(position)
            recyclerView.post { recyclerView.scrollToPosition(position) }
        }
    }

    private fun scrollToEnd() {
        recyclerView.postDelayed({
            val position = adapter!!.itemCount - 1
            linearLayoutManager!!.scrollToPositionWithOffset(position, -9999)
        }, 100)
    }

    private fun handleSendMessage(userData: ChatDataItem) {
        setIsGenerating(true)
        adapter!!.addItem(userData)
        addResponsePlaceholder()
        val input: String
        val hasSessionName = !TextUtils.isEmpty(this.sessionName)
        var sessionName: String? = null
        if (userData.audioUri != null) {
            val audioPath = FileUtils.getPathForUri(userData.audioUri!!)
            if (audioPath == null) {
                Toast.makeText(this, "Audio file not found", Toast.LENGTH_LONG).show()
                return
            }
            if (userData.audioDuration <= 0.1) {
                userData.audioDuration =
                    FileUtils.getAudioDuration(audioPath).toFloat()
            }
            input = String.format("<audio>%s</audio>%s", audioPath, userData.text)
            if (!hasSessionName) {
                sessionName = "[Audio]" + userData.text
            }
        } else if (userData.imageUri != null) {
            val imagePath = FileUtils.getPathForUri(userData.imageUri!!)
            if (imagePath == null) {
                Toast.makeText(this, "image file not found", Toast.LENGTH_LONG).show()
                return
            }
            input = String.format("<img>%s</img>%s", imagePath, userData.text)
            if (!hasSessionName) {
                sessionName = "[Image]" + userData.text
            }
        } else {
            input = userData.text!!
            if (!hasSessionName) {
                sessionName = userData.text
            }
        }
        if (!hasSessionName) {
            chatDataManager!!.addOrUpdateSession(sessionId!!, modelId)
            this.sessionName =
                if (sessionName!!.length > 100) sessionName.substring(0, 100) else sessionName
            chatDataManager!!.updateSessionName(this.sessionId!!, this.sessionName)
        }
        if (ModelUtils.isDiffusionModel(this.modelName)) {
            chatExecutor!!.execute { submitRequest(input) }
        } else {
            chatExecutor!!.execute { submitRequest(input) }
        }
        chatDataManager!!.addChatData(sessionId, userData)

        smoothScrollToBottom()
    }

    private fun addResponsePlaceholder() {
        val holderItem = ChatDataItem(dateFormat!!.format(Date()), ChatViewHolders.ASSISTANT, "")
        holderItem.hasOmniAudio = chatSession.supportOmni
        adapter!!.addItem(holderItem)
        smoothScrollToBottom()
    }

    private fun submitRequest(input: String) {
        isUserScrolling = false
        stopGenerating = false
        val chatDataItem = adapter!!.recentItem
        val benchMarkResult: HashMap<String, Any>
        if (ModelUtils.isDiffusionModel(this.modelName!!)) {
            val diffusionDestPath = FileUtils.generateDestDiffusionFilePath(
                this,
                sessionId!!
            )
            chatDataItem!!.loading = true
            benchMarkResult = chatSession.generateDiffusion(
                input, diffusionDestPath, 20,
                Random(System.currentTimeMillis()).nextInt(), object : ChatSession.GenerateProgressListener {
                    override fun onProgress(progress: String?): Boolean {
                        if ("100" == progress) {
                            chatDataItem.text = getString(R.string.diffusion_generated_message)
                            chatDataItem.imageUri = Uri.parse(diffusionDestPath)
                        } else {
                            chatDataItem.text = getString(R.string.diffusion_generate_progress, progress)
                            chatDataItem.displayText = chatDataItem.text
                        }
                        runOnUiThread { updateAssistantResponse(chatDataItem) }
                        return false
                    }
                }
            )
            chatDataItem.loading = false
        } else {
            chatDataItem!!.loading = true
            val generateResultProcessor: GenerateResultProcessor =
                R1GenerateResultProcessor(
                    getString(R.string.r1_thinking_message),
                    getString(R.string.r1_think_complete_template))
            generateResultProcessor.generateBegin()
            benchMarkResult = chatSession.generate(input, object: ChatSession.GenerateProgressListener {
                override fun onProgress(progress: String?): Boolean {
                    generateResultProcessor.process(progress)
                    chatDataItem.displayText = generateResultProcessor.getDisplayResult()
                    chatDataItem.text = generateResultProcessor.getRawResult()
                    runOnUiThread { updateAssistantResponse(chatDataItem) }
                    if (stopGenerating) {
                        Log.d(TAG, "stopGenerating requested")
                    }
                    return stopGenerating
                }
            })
            chatDataItem.loading = false
        }
        Log.d(TAG, "submitRequest benchMark: $benchMarkResult")
        runOnUiThread {
            chatDataItem.benchmarkInfo = ModelUtils.generateBenchMarkString(benchMarkResult)
            updateAssistantResponse(chatDataItem)
        }
        chatDataManager!!.addChatData(sessionId, chatDataItem)
        runOnUiThread {
            setIsGenerating(false)
        }
    }

    private fun updateAssistantResponse(chatDataItem: ChatDataItem) {
        adapter!!.updateRecentItem(chatDataItem)
        if (!isUserScrolling) {
            scrollToEnd()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        stopGenerating = true
        chatExecutor!!.submit {
            chatSession.reset()
            chatSession.release()
            chatExecutor!!.shutdownNow()
        }
    }

    override fun onStop() {
        super.onStop()
        AudioPlayService.instance!!.destroy()
    }

    val sessionDebugInfo: String
        get() = chatSession.debugInfo

    companion object {
        const val TAG: String = "ChatActivity"
    }
}