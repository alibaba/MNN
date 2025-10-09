// Created by ruoyi.sjd on 2025/5/6.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.chat.input

import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.text.Editable
import android.text.TextUtils
import android.text.TextWatcher
import android.util.Log
import android.view.View
import android.widget.EditText
import android.widget.ImageView
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.chat.ChatActivity
import com.alibaba.mnnllm.android.chat.input.AttachmentPickerModule.AttachmentType
import com.alibaba.mnnllm.android.chat.input.AttachmentPickerModule.ImagePickCallback
import com.alibaba.mnnllm.android.chat.ChatActivity.Companion.TAG
import com.alibaba.mnnllm.android.chat.input.VoiceRecordingModule.VoiceRecordingListener
import com.alibaba.mnnllm.android.chat.chatlist.ChatViewHolders
import com.alibaba.mnnllm.android.chat.model.ChatDataItem
import com.alibaba.mnnllm.android.databinding.ActivityChatBinding
import com.alibaba.mnnllm.android.llm.LlmSession
import com.alibaba.mnnllm.android.utils.KeyboardUtils
import com.alibaba.mnnllm.android.model.ModelUtils
import com.alibaba.mnnllm.android.utils.Permissions.REQUEST_RECORD_AUDIO_PERMISSION
import java.util.Date
import com.alibaba.mnnllm.android.modelist.ModelListManager
import com.alibaba.mnnllm.android.modelsettings.ModelConfig

class ChatInputComponent(
    private val chatActivity: ChatActivity,
    private val binding: ActivityChatBinding,
    modelId:String,
    modelName: String,
) {
    private var currentModelId: String = modelId
    private var currentModelName: String = modelName
    private var onStopGenerating: (() -> Unit)? = null
    private var onThinkingModeChanged: ((Boolean) -> Unit)? = null
    private var onAudioOutputModeChanged: ((Boolean) -> Unit)? = null
    private var onSendMessage: ((ChatDataItem) -> Unit)? = null
    private lateinit var editUserMessage: EditText
    private var buttonSend: ImageView = binding.btnSend
    private lateinit var imageMore: ImageView
    var attachmentPickerModule: AttachmentPickerModule? = null
    private lateinit var voiceRecordingModule: VoiceRecordingModule
    private var currentUserMessage: ChatDataItem? = null
    private var buttonSwitchVoice: View? = null

    init {
        buttonSend.setEnabled(false)
        buttonSend.setOnClickListener { handleSendClick() }
        setupEditText()
        setupAttachmentPickerModule()
        setupVoiceRecordingModule()
        setupThinkingMode()
        setupToggleAudioOutput()
        updateAudioOutput()
    }

    /**
     * Update the model name and refresh related UI components
     */
    fun updateModel(newModelId:String, newModelName: String) {
        val oldModelId = currentModelId
        currentModelId = newModelId
        val oldModelName = currentModelName
        currentModelName = newModelName
        
        // Update thinking mode
        setupThinkingMode()
        
        // Update audio output
        updateAudioOutput()
        
        // Update attachment picker if model capabilities changed
        if (ModelUtils.isVisualModel(oldModelId) != ModelUtils.isVisualModel(newModelId) ||
            ModelUtils.isAudioModel(oldModelId) != ModelUtils.isAudioModel(newModelId)) {
            setupAttachmentPickerModule()
        }
        // Update voice recording module
        voiceRecordingModule.updateModel(newModelName)
        
        // Update voice button visibility
        updateVoiceButtonVisibility()
    }

    private fun setupToggleAudioOutput() {
        binding.btnToggleAudioOutput.setOnClickListener {
            if (chatActivity.isLoading) {
                return@setOnClickListener
            }
            if (!binding.btnToggleAudioOutput.isSelected) {
                android.app.AlertDialog.Builder(chatActivity)
                    .setMessage(R.string.audio_output_confirm)
                    .setPositiveButton(android.R.string.ok) { _, _ ->
                        binding.btnToggleAudioOutput.isSelected = true
                        onAudioOutputModeChanged?.apply {
                            this(binding.btnToggleAudioOutput.isSelected)
                        }
                    }
                    .setNegativeButton(android.R.string.cancel, null)
                    .show()
            } else {
                binding.btnToggleAudioOutput.isSelected = false
                onAudioOutputModeChanged?.apply {
                    this(binding.btnToggleAudioOutput.isSelected)
                }
            }
        }
        updateAudioOutput()
    }

    private fun updateAudioOutput() {
        if (ModelUtils.supportAudioOutput(currentModelName)) {
            binding.btnToggleAudioOutput.visibility = View.VISIBLE
        } else {
            binding.btnToggleAudioOutput.visibility = View.GONE
        }
    }
    
    private fun setupThinkingMode() {
        val extraTags = ModelListManager.getExtraTags(currentModelId)
        binding.btnToggleThinking.visibility = if (ModelUtils.isSupportThinkingSwitchByTags(extraTags)) {
            binding.btnToggleThinking.isSelected = ModelConfig.loadConfig(currentModelId)?.jinja?.context?.enableThinking != false
            View.VISIBLE
        } else  {
            View.GONE
        }
        binding.btnToggleThinking.setOnClickListener {
            Log.d(TAG, "handleSendClick isGenerating : ${chatActivity.isLoading}")
            if (chatActivity.isLoading) {
                return@setOnClickListener
            }
            binding.btnToggleThinking.isSelected = !binding.btnToggleThinking.isSelected
            onThinkingModeChanged?.apply {
                this(binding.btnToggleThinking.isSelected)
            }
        }
    }

    private fun handleSendClick() {
        Log.d(
            TAG,
            "handleSendClick isGenerating : ${chatActivity.isGenerating}"
        )
        if (chatActivity.isGenerating) {
            this.onStopGenerating?.invoke()
        } else {
            sendUserMessage()
        }
    }

    private fun setupEditText() {
        editUserMessage = binding.etMessage
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

    fun updateSenderButton() {
        var enabled = true
        if (chatActivity.isLoading) {
            enabled = false
        } else if (currentUserMessage == null && TextUtils.isEmpty(editUserMessage.text.toString())) {
            enabled = false
        }
        if (chatActivity.isGenerating) {
            enabled = true
        }
        buttonSend.isEnabled = enabled
        buttonSend.setImageResource(if (!chatActivity.isGenerating) R.drawable.button_send else R.drawable.ic_stop)
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
        currentUserMessage!!.time = chatActivity.dateFormat!!.format(Date())
        editUserMessage.setText("")
        KeyboardUtils.hideKeyboard(editUserMessage)
        this.onSendMessage?.let { it(currentUserMessage!!) }
        if (attachmentPickerModule != null) {
            attachmentPickerModule!!.clearInput()
            attachmentPickerModule!!.hideAttachmentLayout()
        }
        currentUserMessage = null
    }

    private fun updateVoiceButtonVisibility() {
        if (!ModelUtils.isAudioModel(currentModelName)) {
            return
        }
        var visible = true
        if (!ModelUtils.isAudioModel(currentModelName)) {
            visible = false
        } else if (chatActivity.isGenerating) {
            visible = false
        } else if (currentUserMessage != null) {
            visible = false
        } else if (!TextUtils.isEmpty(editUserMessage.text.toString())) {
            visible = false
        }
        buttonSwitchVoice!!.visibility =
            if (visible) View.VISIBLE else View.GONE
    }

    private fun setupAttachmentPickerModule() {
        imageMore = binding.btPlus
        buttonSwitchVoice = binding.btSwitchAudio
        if (!ModelUtils.isVisualModel(currentModelId) && !ModelUtils.isAudioModel(currentModelId)) {
            imageMore.setVisibility(View.GONE)
            return
        }
        attachmentPickerModule = AttachmentPickerModule(chatActivity)
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
            voiceRecordingModule.exitRecordingMode()
            attachmentPickerModule!!.toggleAttachmentVisibility()
        }
    }

    private fun setupVoiceRecordingModule() {
        voiceRecordingModule = VoiceRecordingModule(chatActivity)
        voiceRecordingModule.setOnVoiceRecordingListener(object : VoiceRecordingListener {
            override fun onEnterRecordingMode() {
                updateAudioOutput()
                binding.btnToggleThinking.visibility = View.GONE
                editUserMessage.visibility = View.GONE
                KeyboardUtils.hideKeyboard(editUserMessage)
                if (attachmentPickerModule != null) {
                    attachmentPickerModule!!.hideAttachmentLayout()
                }
                editUserMessage.visibility = View.GONE
            }

            override fun onLeaveRecordingMode() {
                val extraTags = ModelListManager.getExtraTags(currentModelId)
                if (ModelUtils.isSupportThinkingSwitchByTags(extraTags)) {
                    binding.btnToggleThinking.visibility = View.VISIBLE
                }
                updateAudioOutput()
                binding.btnSend.visibility = View.VISIBLE
                editUserMessage.visibility = View.VISIBLE
                editUserMessage.requestFocus()
                KeyboardUtils.showKeyboard(editUserMessage)
            }

            override fun onRecordSuccess(duration: Float, recordingFilePath: String?) {
                val chatDataItem = ChatDataItem.createAudioInputData(
                    chatActivity.dateFormat!!.format(Date()),
                    "",
                    recordingFilePath!!,
                    duration
                )
                this@ChatInputComponent.onSendMessage?.let { it(chatDataItem) }
            }

            override fun onRecordCanceled() {
            }
        })
        voiceRecordingModule!!.setup(chatActivity.isAudioModel)
    }

    fun setOnSendMessage(onSendMessage: (ChatDataItem)->Unit) {
        this.onSendMessage = onSendMessage
    }

    fun setOnThinkingModeChanged(onThinkingModeChanged: (Boolean)->Unit) {
        this.onThinkingModeChanged = onThinkingModeChanged
    }

    fun setOnAudioOutputModeChanged(onAudioOutputChanged: (Boolean)->Unit) {
        this.onAudioOutputModeChanged = onAudioOutputChanged
    }

    fun setIsGenerating(generating: Boolean) {
        updateSenderButton()
        updateVoiceButtonVisibility()
    }

    fun handleResult(requestCode: Int, resultCode: Int, data: Intent?) {
        if (attachmentPickerModule != null && attachmentPickerModule!!.canHandleResult(requestCode)) {
            attachmentPickerModule?.onActivityResult(requestCode, resultCode, data)
        }
    }

    fun onLoadingStatesChanged(loading: Boolean) {
        this.updateSenderButton()
        if (!loading && ModelUtils.isAudioModel(currentModelName)) {
            voiceRecordingModule.onEnabled()
        }

    }

    fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        if (requestCode == REQUEST_RECORD_AUDIO_PERMISSION) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                voiceRecordingModule.handlePermissionAllowed()
            } else {
                voiceRecordingModule.handlePermissionDenied()
            }
        } else if (attachmentPickerModule != null && 
                   requestCode == AttachmentPickerModule.REQUEST_CODE_CAMERA_PERMISSION) {
            attachmentPickerModule!!.onRequestPermissionsResult(requestCode, permissions, grantResults)
        }
    }

    fun setOnStopGenerating(onStopGenerating: () -> Unit) {
        this.onStopGenerating = onStopGenerating
    }

}