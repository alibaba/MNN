// Created by ruoyi.sjd on 2025/06/18.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.chat.voice

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.chat.ChatActivity
import com.alibaba.mnnllm.android.chat.ChatPresenter
import com.alibaba.mnnllm.android.databinding.FragmentVoiceChatBinding
import com.alibaba.mnnllm.android.utils.KeyboardUtils

class VoiceChatFragment : Fragment(), VoiceChatView {
    
    companion object {
        const val TAG = "VoiceChatFragment"
        private const val ARG_MODEL_NAME = "model_name"
        private const val ARG_MODEL_ID = "model_id"
        private const val ARG_CHAT_PRESENTER = "chat_presenter"
        
        fun newInstance(modelName: String, modelId: String, chatPresenter: ChatPresenter): VoiceChatFragment {
            val fragment = VoiceChatFragment()
            fragment.chatPresenter = chatPresenter
            val args = Bundle()
            args.putString(ARG_MODEL_NAME, modelName)
            args.putString(ARG_MODEL_ID, modelId)
            fragment.arguments = args
            return fragment
        }
    }

    private var statusText: String = ""
    private var _binding: FragmentVoiceChatBinding? = null
    private val binding get() = _binding!!
    
    private var modelName: String = ""
    private var isTranscriptVisible = true
    private lateinit var chatPresenter: ChatPresenter

    // UI components
    private lateinit var transcriptAdapter: VoiceTranscriptAdapter

    // Presenter
    private var presenter: VoiceChatPresenter? = null


    // Permissions
    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted: Boolean ->
            if (isGranted) {
                Log.d(TAG, "RECORD_AUDIO permission granted")
                startPresenter()
            } else {
                Log.w(TAG, "RECORD_AUDIO permission denied")
                Toast.makeText(requireContext(), R.string.voice_chat_permission_required, Toast.LENGTH_LONG).show()
                endVoiceChatSession(false)
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        arguments?.let {
            modelName = it.getString(ARG_MODEL_NAME, "")
        }
        Log.d(TAG, "onCreate: modelName=$modelName")
    }
    
    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentVoiceChatBinding.inflate(inflater, container, false)
        return binding.root
    }
    
    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        Log.d(TAG, "onViewCreated")
        
        setupToolbar()
        setupClickListeners()
        setupRecyclerView()
        startVoiceChatSession()
        KeyboardUtils.hideKeyboard(_binding!!.root)
    }
    
    private fun setupToolbar() {
        binding.toolbar.title = modelName
        binding.toolbar.setNavigationOnClickListener {
            endVoiceChatSession(true)
        }
        
        // Handle menu item clicks
        binding.toolbar.setOnMenuItemClickListener { menuItem ->
            when (menuItem.itemId) {
                R.id.action_voice_settings -> {
                    openVoiceModelSettings()
                    true
                }
                else -> false
            }
        }
    }
    
    private fun setupClickListeners() {
        binding.buttonEndCall.setOnClickListener {
            endVoiceChatSession(true)
        }
        
        binding.buttonToggleText.setOnClickListener {
            toggleTranscriptVisibility()
        }
        
        binding.tvVoiceChatStatus.setOnClickListener {
            presenter?.stopGeneration()
        }
    }
    
    private fun setupRecyclerView() {
        transcriptAdapter = VoiceTranscriptAdapter(mutableListOf())
        binding.rvVoiceTranscript.layoutManager = LinearLayoutManager(requireContext())
        binding.rvVoiceTranscript.adapter = transcriptAdapter
        Log.d(TAG, "RecyclerView set up")
    }
    
    private fun startVoiceChatSession() {
        Log.i(TAG, "Starting voice chat session...")
        updateStatus(VoiceChatState.CONNECTING)
        checkAndRequestPermission()
    }

    private fun checkAndRequestPermission() {
        when {
            ContextCompat.checkSelfPermission(
                requireContext(),
                Manifest.permission.RECORD_AUDIO
            ) == PackageManager.PERMISSION_GRANTED -> {
                Log.d(TAG, "RECORD_AUDIO permission already granted")
                startPresenter()
            }
            else -> {
                Log.d(TAG, "Requesting RECORD_AUDIO permission")
                requestPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
            }
        }
    }
    
    private fun startPresenter() {
        presenter = VoiceChatPresenter(
            requireActivity(), 
            this, 
            chatPresenter,
            viewLifecycleOwner.lifecycleScope
        )
        presenter?.start()
    }

    private fun endVoiceChatSession(showToast: Boolean) {
        Log.i(TAG, "Ending voice chat session...")
        presenter?.stop()
        presenter = null
        
        if (showToast) {
            Toast.makeText(context, R.string.voice_chat_session_ended, Toast.LENGTH_SHORT).show()
        }
        
        // Return to ChatActivity, which will automatically show the voice chat history
        activity?.supportFragmentManager?.popBackStack()
    }
    
    private fun toggleTranscriptVisibility() {
        isTranscriptVisible = !isTranscriptVisible
//        binding.buttonToggleTextImage.setImageResource(
//            if (isTranscriptVisible) R.drawable.text_on else R.drawable.text_off
//        )
        // Use the current state to update visibility
        val currentState = when {
            binding.layoutVoiceLoading.visibility == View.VISIBLE -> VoiceChatState.CONNECTING
            binding.rvVoiceTranscript.visibility == View.VISIBLE -> VoiceChatState.PROCESSING
            else -> VoiceChatState.LISTENING
        }
        updateStatus(currentState)
        Log.d(TAG, "Transcript visibility toggled: $isTranscriptVisible")
    }

    // VoiceChatView implementation
    override fun updateStatus(state: VoiceChatState) {
        if (_binding == null) return // Guard against null binding
        
        statusText = when (state) {
            VoiceChatState.CONNECTING -> getString(R.string.voice_chat_connecting)
            VoiceChatState.GREETING -> getString(R.string.voice_chat_greeting)
            VoiceChatState.LISTENING -> getString(R.string.voice_chat_listening)
            VoiceChatState.PROCESSING -> getString(R.string.voice_chat_stop) // Show "Stop" when processing
            VoiceChatState.THINKING -> getString(R.string.voice_chat_thinking)
            VoiceChatState.SPEAKING -> getString(R.string.voice_chat_stop) // Show "Stop" when speaking
            VoiceChatState.STOPPING -> getString(R.string.voice_chat_stopping)
            VoiceChatState.ERROR -> getString(R.string.voice_chat_error)
        }
        Log.d(TAG, "updateStatus: state=$state, statusText=$statusText")
        val isLoading = state == VoiceChatState.CONNECTING
        binding.layoutVoiceLoading.visibility = if (isLoading) View.VISIBLE else View.GONE
        updateViewVisibility()
    }

    override fun addTranscript(transcript: Transcript) {
        if (_binding == null) return
        
        updateViewVisibility()
        transcriptAdapter.addTranscript(transcript)
        binding.rvVoiceTranscript.scrollToPosition(transcriptAdapter.itemCount - 1)
    }

    override fun updateLastTranscript(text: String) {
        if (_binding == null) return // Guard against null binding
        
        transcriptAdapter.updateLastTranscript(text)
        binding.rvVoiceTranscript.scrollToPosition(transcriptAdapter.itemCount - 1)
    }

    override fun showError(message: String) {
        if (context == null) return // Guard against null context
        
        Toast.makeText(context, message, Toast.LENGTH_LONG).show()
        updateStatus(VoiceChatState.ERROR)
    }
    
    private fun updateViewVisibility() {
        if (_binding == null) return

        // Update transcript visibility based on both showTranscript flag and isTranscriptVisible preference
        binding.rvVoiceTranscript.visibility = if (isTranscriptVisible) View.VISIBLE else View.GONE
        
        // Always show status text at bottom
        binding.tvVoiceChatStatus.visibility = View.VISIBLE
        binding.tvVoiceChatStatus.text = statusText
        // Make it clickable when showing "Stop"
        val stopText = getString(R.string.voice_chat_stop)
        binding.tvVoiceChatStatus.isClickable = statusText == stopText
    }

    override fun onDestroyView() {
        super.onDestroyView()
        Log.d(TAG, "onDestroyView")
        presenter?.stop()
        _binding = null
    }

    override fun stopGeneration() {
        // This method is called by the presenter, we don't need to implement it here
        // The presenter handles the actual stopping logic
    }

    private fun openVoiceModelSettings() {
        try {
            val voiceModelMarketBottomSheet = VoiceModelMarketBottomSheet.newInstance()
            
            // Set up model change callback
            voiceModelMarketBottomSheet.setOnModelChangedCallback { voiceModelType, modelId ->
                Log.d(TAG, "Voice model changed: $voiceModelType, modelId: $modelId")
                
                // Recreate voice services when model changes
                presenter?.recreateVoiceServices()
                
                // Show toast to user
                val modelTypeName = when (voiceModelType) {
                    VoiceModelMarketBottomSheet.VoiceModelType.TTS -> getString(R.string.tts_model_name)
                    VoiceModelMarketBottomSheet.VoiceModelType.ASR -> getString(R.string.asr_model_name)
                }
                Toast.makeText(requireContext(), 
                    getString(R.string.voice_model_changed_recreating_services, modelTypeName), 
                    Toast.LENGTH_SHORT).show()
            }
            
            voiceModelMarketBottomSheet.show(childFragmentManager, "voice_model_market")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to show voice model market", e)
            Toast.makeText(context, getString(R.string.failed_to_show_voice_model_market, e.message), Toast.LENGTH_LONG).show()
        }
    }
    
    override fun showGreetingMessage() {
        if (_binding == null) return
        
        // Add greeting message to transcript display (but don't save to DB or chat history)
        val greetingMessage = getString(R.string.voice_chat_ready_greeting)
        val greetingTranscript = Transcript(isUser = false, text = greetingMessage)
        transcriptAdapter.addTranscript(greetingTranscript)
        binding.rvVoiceTranscript.scrollToPosition(transcriptAdapter.itemCount - 1)
        
        Log.d(TAG, "Greeting message shown: $greetingMessage")
    }
} 