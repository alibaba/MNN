// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.chat

import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.text.Editable
import android.text.TextUtils
import android.text.TextWatcher
import android.util.Log
import android.view.KeyEvent
import android.view.Menu
import android.view.MenuItem
import android.view.View
import android.view.WindowInsets
import android.widget.EditText
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.widget.NestedScrollView
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.alibaba.mnnllm.android.ChatService
import com.alibaba.mnnllm.android.ChatSession
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.audio.AudioPlayer
import com.alibaba.mnnllm.android.chat.AttachmentPickerModule.AttachmentType
import com.alibaba.mnnllm.android.chat.AttachmentPickerModule.ImagePickCallback
import com.alibaba.mnnllm.android.chat.GenerateResultProcessor.R1GenerateResultProcessor
import com.alibaba.mnnllm.android.chat.VoiceRecordingModule.VoiceRecordingListener
import com.alibaba.mnnllm.android.databinding.ActivityChatBinding
import com.alibaba.mnnllm.android.modelsettings.SettingsBottomSheetFragment
import com.alibaba.mnnllm.android.utils.AudioPlayService
import com.alibaba.mnnllm.android.utils.FileUtils
import com.alibaba.mnnllm.android.utils.KeyboardUtils
import com.alibaba.mnnllm.android.utils.ModelPreferences
import com.alibaba.mnnllm.android.utils.ModelUtils
import com.alibaba.mnnllm.android.utils.Permissions.REQUEST_RECORD_AUDIO_PERMISSION
import com.alibaba.mnnllm.android.utils.PreferenceUtils
import com.google.android.material.bottomsheet.BottomSheetBehavior
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
    private val _isGenerating = MutableStateFlow(false)
    private var isGenerating: Boolean
        get() = _isGenerating.value
        set(value) {
            _isGenerating.value = value
        }
    private lateinit var recyclerView: RecyclerView
    private var adapter: ChatRecyclerViewAdapter? = null
    private lateinit var editUserMessage: EditText
    private lateinit var buttonSend: ImageView

    private lateinit var imageMore: ImageView
    private var layoutModelLoading: View? = null

    private var dateFormat: DateFormat? = null
    private lateinit var chatSession: ChatSession

    var sessionId: String? = null
        private set

    var modelName: String? = null
        private set
    private var modelId: String? = null

    private var chatExecutor: ScheduledExecutorService? = null

    private var linearLayoutManager: LinearLayoutManager? = null

    private var chatDataManager: ChatDataManager? = null

    private var isUserScrolling = false

    private var voiceRecordingModule: VoiceRecordingModule? = null

    private var isAudioModel = false
    private var attachmentPickerModule: AttachmentPickerModule? = null
    private var buttonSwitchVoice: View? = null

    private var currentUserMessage: ChatDataItem? = null

    private var isLoading = false
    private var sessionName: String? = null
    private var stopGenerating = false
    private val CONFIG_SHOW_CUSTOM_TOOLBAR = false
    private lateinit var binding: ActivityChatBinding
    private lateinit var bottomSheetBehavior: BottomSheetBehavior<NestedScrollView>
    private var audioPlayer: AudioPlayer? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityChatBinding.inflate(layoutInflater)
        setContentView(binding.root)
        val toolbar = binding.toolbar
        setSupportActionBar(toolbar)
        modelName = intent.getStringExtra("modelName")
        modelId = intent.getStringExtra("modelId")
        layoutModelLoading = findViewById(R.id.layout_model_loading)
        if (supportActionBar != null) {
            supportActionBar!!.setDisplayHomeAsUpEnabled(true)
            supportActionBar!!.setDisplayShowTitleEnabled(!CONFIG_SHOW_CUSTOM_TOOLBAR)
            supportActionBar!!.title = getString(R.string.app_name)
        }
        binding.btnToggleThinking.visibility = if (ModelUtils.isSupportThinkingSwitch(modelName!!)) {
                binding.btnToggleThinking.isSelected = true
                View.VISIBLE
            } else  {
                View.GONE
            }
        binding.btnToggleThinking.setOnClickListener {
            binding.btnToggleThinking.isSelected = !binding.btnToggleThinking.isSelected
            chatSession.updateAssistantPrompt(if (binding.btnToggleThinking.isSelected) {
                "<|im_start|>assistant\n%s<|im_end|>\n"
            } else {
                "<|im_start|>assistant\n<think>\n</think>%s<|im_end|>\n"
            })
        }
        chatExecutor = Executors.newScheduledThreadPool(1)
        chatDataManager = ChatDataManager.getInstance(this)
        this.setupSession()
        dateFormat = SimpleDateFormat("hh:mm aa", Locale.getDefault())
        this.setupRecyclerView()
        setupEditText()
        buttonSend = binding.btnSend
        buttonSend.setEnabled(false)
        buttonSend.setOnClickListener { handleSendClick() }
        isAudioModel = ModelUtils.isAudioModel(modelName!!)
        setupVoiceRecordingModule()
        setupAttachmentPickerModule()
        smoothScrollToBottom()
        setupBottomSheetBehavior()
    }

    private fun setupBottomSheetBehavior() {
        bottomSheetBehavior = BottomSheetBehavior.from(binding.bottomSheet)
        bottomSheetBehavior.state = BottomSheetBehavior.STATE_HIDDEN
    }

    private fun handleSendClick() {
        Log.d(
            TAG,
            "handleSendClick isGenerating : $isGenerating"
        )
        if (isGenerating) {
            stopGenerating = true
        } else {
            sendUserMessage()
        }
    }

    private fun setupSession() {
        val chatService = ChatService.provide()
        sessionId = intent.getStringExtra("chatSessionId")
        val chatDataItemList: List<ChatDataItem>?
        if (!TextUtils.isEmpty(sessionId)) {
            chatDataItemList = chatDataManager!!.getChatDataBySession(sessionId!!)
            if (!chatDataItemList.isNullOrEmpty()) {
                sessionName = chatDataItemList[0].text
            }
        } else {
            chatDataItemList = null
        }
        if (ModelUtils.isDiffusionModel(modelName!!)) {
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
                ModelUtils.isOmni(modelName!!)
            )

        }
        sessionId = chatSession.sessionId
        chatSession.setKeepHistory(
            !ModelUtils.isVisualModel(modelName!!) && !ModelUtils.isAudioModel(
                modelName!!
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
            if (!loading && voiceRecordingModule != null && ModelUtils.isAudioModel(modelName!!)) {
                voiceRecordingModule!!.onEnabled()
            }
            updateSenderButton()
            layoutModelLoading!!.visibility =
                if (loading) View.VISIBLE else View.GONE
            if (supportActionBar != null) {
                supportActionBar!!.setDisplayHomeAsUpEnabled(true)
                if (CONFIG_SHOW_CUSTOM_TOOLBAR) {
//                    toolbarTitle.visibility = View.VISIBLE
//                    toolbarTitle.text =
//                        if (loading) getString(R.string.model_loading) else getString(R.string.app_name)
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
//        ViewCompat.setOnApplyWindowInsetsListener(recyclerView) { view, insets ->
//            val bottomInset = insets.systemWindowInsetBottom
//            view.setPadding(view.paddingLeft, view.paddingTop, view.paddingRight, bottomInset)
//            insets.consumeSystemWindowInsets()
//        }
        adapter = ChatRecyclerViewAdapter(this, initData(), this.modelName!!)
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

    private fun setupEditText() {
        editUserMessage = binding.etMessage
        editUserMessage.setOnEditorActionListener { v: TextView?, actionId: Int, event: KeyEvent? ->
            if ((event != null && event.keyCode == KeyEvent.KEYCODE_ENTER && event.action == KeyEvent.ACTION_DOWN)) {
                Log.d(
                    TAG,
                    "onEditorAction" + actionId + "  getAction: " + event.action + "code: " + event.keyCode
                )
                sendUserMessage()
                return@setOnEditorActionListener true
            }
            false
        }
        editUserMessage.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence, start: Int, count: Int, after: Int) {
            }

            override fun onTextChanged(s: CharSequence, start: Int, before: Int, count: Int) {
            }

            override fun afterTextChanged(s: Editable) {
                updateSenderButton()
                updateVoiceButtonVisibility()
            }
        })
    }

    private fun initData(): MutableList<ChatDataItem> {
        val data: MutableList<ChatDataItem> = ArrayList()
        data.add(ChatDataItem(dateFormat!!.format(Date()), ChatViewHolders.HEADER, ""))
        data.add(
            ChatDataItem(
                dateFormat!!.format(Date()), ChatViewHolders.ASSISTANT,
                getString(
                    if (ModelUtils.isDiffusionModel(modelName!!))
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

    private fun setupAttachmentPickerModule() {
        imageMore = findViewById(R.id.bt_plus)
        buttonSwitchVoice = findViewById(R.id.bt_switch_audio)
        if (!ModelUtils.isVisualModel(this.modelName!!) && !ModelUtils.isAudioModel(this.modelName!!)) {
            imageMore.setVisibility(View.GONE)
            return
        }
        attachmentPickerModule = AttachmentPickerModule(this)
        attachmentPickerModule!!.setOnImagePickCallback(object : ImagePickCallback {
            override fun onAttachmentPicked(imageUri: Uri?, audio: AttachmentType?) {
                imageMore.setVisibility(View.GONE)
                updateVoiceButtonVisibility()
                currentUserMessage = ChatDataItem(ChatViewHolders.USER)
                if (audio == AttachmentType.Audio) {
                    currentUserMessage!!.audioUri = imageUri
                } else {
                    currentUserMessage!!.imageUri = imageUri
                }
                updateSenderButton()
            }

            override fun onAttachmentRemoved() {
                currentUserMessage = null
                imageMore.setVisibility(View.VISIBLE)
                updateSenderButton()
                updateVoiceButtonVisibility()
            }

            override fun onAttachmentLayoutShow() {
                imageMore.setImageResource(R.drawable.ic_bottom)
            }

            override fun onAttachmentLayoutHide() {
                imageMore.setImageResource(R.drawable.ic_plus)
            }
        })
        imageMore.setOnClickListener {
            if (voiceRecordingModule != null) {
                voiceRecordingModule?.exitRecordingMode()
            }
            attachmentPickerModule!!.toggleAttachmentVisibility()
        }
    }

    private fun updateVoiceButtonVisibility() {
        if (!ModelUtils.isAudioModel(modelName!!)) {
            return
        }
        var visible = true
        if (!ModelUtils.isAudioModel(modelName!!)) {
            visible = false
        } else if (isGenerating) {
            visible = false
        } else if (currentUserMessage != null) {
            visible = false
        } else if (!TextUtils.isEmpty(editUserMessage.text.toString())) {
            visible = false
        }
        buttonSwitchVoice!!.visibility =
            if (visible) View.VISIBLE else View.GONE
    }

    private fun updateSenderButton() {
        var enabled = true
        if (isLoading) {
            enabled = false
        } else if (currentUserMessage == null && TextUtils.isEmpty(editUserMessage.text.toString())) {
            enabled = false
        }
        if (isGenerating) {
            enabled = true
        }
        buttonSend.isEnabled = enabled
        buttonSend.setImageResource(if (!isGenerating) R.drawable.button_send else R.drawable.ic_stop)
    }

    private fun setupVoiceRecordingModule() {
        voiceRecordingModule = VoiceRecordingModule(this)
        voiceRecordingModule!!.setOnVoiceRecordingListener(object : VoiceRecordingListener {
            override fun onEnterRecordingMode() {
                binding.btnToggleThinking.visibility = View.GONE
                editUserMessage.visibility = View.GONE
                KeyboardUtils.hideKeyboard(editUserMessage)
                if (attachmentPickerModule != null) {
                    attachmentPickerModule!!.hideAttachmentLayout()
                }
                editUserMessage.visibility = View.GONE
            }

            override fun onLeaveRecordingMode() {
                if (ModelUtils.isSupportThinkingSwitch(modelName!!)) {
                    binding.btnToggleThinking.visibility = View.VISIBLE
                }
                binding.btnSend.visibility = View.VISIBLE
                editUserMessage.visibility = View.VISIBLE
                editUserMessage.requestFocus()
                KeyboardUtils.showKeyboard(editUserMessage)
            }

            override fun onRecordSuccess(duration: Float, recordingFilePath: String?) {
                val chatDataItem = ChatDataItem.createAudioInputData(
                    dateFormat!!.format(Date()),
                    "",
                    recordingFilePath!!,
                    duration
                )
                handleSendMessage(chatDataItem)
            }

            override fun onRecordCanceled() {
            }
        })
        voiceRecordingModule!!.setup(isAudioModel)
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

        if (requestCode == REQUEST_RECORD_AUDIO_PERMISSION) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                voiceRecordingModule!!.handlePermissionAllowed()
            } else {
                voiceRecordingModule!!.handlePermissionDenied()
            }
        }
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
        updateSenderButton()
        updateVoiceButtonVisibility()
    }

    @Deprecated("This method has been deprecated in favor of using the Activity Result API\n      which brings increased type safety via an {@link ActivityResultContract} and the prebuilt\n      contracts for common intents available in\n      {@link androidx.activity.result.contract.ActivityResultContracts}, provides hooks for\n      testing, and allow receiving results in separate, testable classes independent from your\n      activity. Use\n      {@link #registerForActivityResult(ActivityResultContract, ActivityResultCallback)}\n      with the appropriate {@link ActivityResultContract} and handling the result in the\n      {@link ActivityResultCallback#onActivityResult(Object) callback}.")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (attachmentPickerModule != null && attachmentPickerModule!!.canHandleResult(requestCode)) {
            attachmentPickerModule?.onActivityResult(requestCode, resultCode, data!!)
        }
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

    private fun sendUserMessage() {
        if (!buttonSend.isEnabled) {
            return
        }
        val inputString = editUserMessage.text.toString().trim { it <= ' ' }
        if (currentUserMessage == null) {
            currentUserMessage = ChatDataItem(ChatViewHolders.USER)
        }
        currentUserMessage!!.text = inputString
        currentUserMessage!!.time = dateFormat!!.format(Date())
        handleSendMessage(currentUserMessage!!)
        currentUserMessage = null
    }

    private fun createUserMessage(text:String):ChatDataItem {
        val userMessage = ChatDataItem(ChatViewHolders.USER)
        userMessage.text = text
        userMessage.time = dateFormat!!.format(Date())
        return userMessage
    }

    private fun handleSendMessage(userData: ChatDataItem) {
        setIsGenerating(true)
        editUserMessage.setText("")
        adapter!!.addItem(userData)
        addResponsePlaceholder()
        val input: String
        val hasSessionName = !TextUtils.isEmpty(this.sessionName)
        var sessionName: String? = null
        if (userData.audioUri != null) {
            val audioPath = attachmentPickerModule!!.getPathForUri(userData.audioUri!!)
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
            val imagePath = attachmentPickerModule!!.getPathForUri(userData.imageUri!!)
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
        if (ModelUtils.isDiffusionModel(this.modelName!!)) {
            chatExecutor!!.execute { submitRequest(input) }
        } else {
            chatExecutor!!.execute { submitRequest(input) }
        }
        chatDataManager!!.addChatData(sessionId, userData)
        if (attachmentPickerModule != null) {
            attachmentPickerModule!!.clearInput()
        }
        smoothScrollToBottom()
        KeyboardUtils.hideKeyboard(editUserMessage)
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