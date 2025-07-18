// Created by ruoyi.sjd on 2025/01/17.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.chat.voice

import android.app.Dialog
import android.content.DialogInterface
import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.core.content.ContextCompat
import androidx.lifecycle.ViewModelProvider
import androidx.recyclerview.widget.LinearLayoutManager
import com.alibaba.mls.api.download.DownloadState
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.mainsettings.MainSettings
import com.alibaba.mnnllm.android.databinding.BottomSheetVoiceModelMarketBinding
import com.alibaba.mnnllm.android.modelmarket.ModelMarketAdapter
import com.alibaba.mnnllm.android.modelmarket.ModelMarketItemListener
import com.alibaba.mnnllm.android.modelmarket.ModelMarketItemWrapper
import com.google.android.material.bottomsheet.BottomSheetBehavior
import com.google.android.material.bottomsheet.BottomSheetDialog
import com.google.android.material.bottomsheet.BottomSheetDialogFragment

class VoiceModelMarketBottomSheet : BottomSheetDialogFragment(), ModelMarketItemListener {
    
    private var _binding: BottomSheetVoiceModelMarketBinding? = null
    private val binding get() = _binding!!
    
    private lateinit var viewModel: VoiceModelMarketViewModel
    private lateinit var adapter: ModelMarketAdapter
    
    private var currentModelType: VoiceModelType = VoiceModelType.TTS
    private var onDismissCallback: (() -> Unit)? = null
    private var onModelChangedCallback: ((VoiceModelType, String) -> Unit)? = null
    
    enum class VoiceModelType(val displayNameRes: Int) {
        TTS(R.string.tts_models),
        ASR(R.string.asr_models)
    }

    override fun onCreateDialog(savedInstanceState: Bundle?): Dialog {
        Log.d(TAG, "onCreateDialog called")
        try {
            val dialog = super.onCreateDialog(savedInstanceState) as BottomSheetDialog
            Log.d(TAG, "BottomSheetDialog created")
            
            dialog.setOnShowListener { dialogInterface ->
                Log.d(TAG, "BottomSheetDialog onShow triggered")
                val bottomSheetDialog = dialogInterface as BottomSheetDialog
                val bottomSheet = bottomSheetDialog.findViewById<View>(com.google.android.material.R.id.design_bottom_sheet)
                bottomSheet?.let {
                    val behavior = BottomSheetBehavior.from(it)
                    behavior.state = BottomSheetBehavior.STATE_EXPANDED
                    behavior.peekHeight = (resources.displayMetrics.heightPixels * 0.8).toInt()
                    Log.d(TAG, "BottomSheet behavior configured")
                }
            }
            Log.d(TAG, "onCreateDialog completed successfully")
            return dialog
        } catch (e: Exception) {
            Log.e(TAG, "Error in onCreateDialog", e)
            throw e
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        Log.d(TAG, "onCreateView called")
        try {
            _binding = BottomSheetVoiceModelMarketBinding.inflate(inflater, container, false)
            Log.d(TAG, "Binding inflated successfully")
            return binding.root
        } catch (e: Exception) {
            Log.e(TAG, "Error in onCreateView", e)
            throw e
        }
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        Log.d(TAG, "onViewCreated called")
        try {
            super.onViewCreated(view, savedInstanceState)
            
            setupViewModel()
            setupRecyclerView()
            setupSegmentControl()
            setupCloseButton()
            setupMessage()
            
            // Load initial data
            loadModelsForType(currentModelType)
            Log.d(TAG, "onViewCreated completed successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Error in onViewCreated", e)
        }
    }
    
    private fun setupViewModel() {
        viewModel = ViewModelProvider(this)[VoiceModelMarketViewModel::class.java]
        
        // Observe models
        viewModel.models.observe(viewLifecycleOwner) { models ->
            adapter.submitList(models)
            updateEmptyState(models.isEmpty())
        }
        
        // Observe progress updates
        viewModel.progressUpdate.observe(viewLifecycleOwner) { (modelId, downloadInfo) ->
            adapter.updateProgress(modelId, downloadInfo)
        }
        
        // Observe item updates
        viewModel.itemUpdate.observe(viewLifecycleOwner) { modelId ->
            adapter.updateItem(modelId)
            
            // Check if model was just downloaded and set as default if needed
            handleModelDownloadComplete(modelId)
        }
    }
    
    private fun setupRecyclerView() {
        adapter = ModelMarketAdapter(this)
        binding.recyclerView.layoutManager = LinearLayoutManager(requireContext())
        binding.recyclerView.adapter = adapter
        
        // Set up the voice model change callback
        adapter.setVoiceModelChangedCallback { voiceModelType, modelId ->
            // Convert to VoiceModelMarketBottomSheet.VoiceModelType
            val bottomSheetVoiceModelType = when (voiceModelType) {
                com.alibaba.mnnllm.android.modelmarket.MarketHolderVoiceDelegate.VoiceModelType.TTS -> VoiceModelType.TTS
                com.alibaba.mnnllm.android.modelmarket.MarketHolderVoiceDelegate.VoiceModelType.ASR -> VoiceModelType.ASR
                else -> return@setVoiceModelChangedCallback
            }
            
            Log.d(TAG, "Voice model changed: $voiceModelType, modelId: $modelId")
            onModelChangedCallback?.invoke(bottomSheetVoiceModelType, modelId)
        }
    }
    
    private fun setupSegmentControl() {
        binding.segmentTts.setOnClickListener {
            if (currentModelType != VoiceModelType.TTS) {
                switchToModelType(VoiceModelType.TTS)
            }
        }
        
        binding.segmentAsr.setOnClickListener {
            if (currentModelType != VoiceModelType.ASR) {
                switchToModelType(VoiceModelType.ASR)
            }
        }
        
        // Set initial state
        updateSegmentControl()
    }
    
    private fun setupCloseButton() {
        binding.closeButton.setOnClickListener {
            dismiss()
        }
    }
    
    private fun setupMessage() {
        val message = arguments?.getString(ARG_MESSAGE)
        if (!message.isNullOrEmpty()) {
            binding.messageContainer.visibility = View.VISIBLE
            binding.messageText.text = message
            Log.d(TAG, "Message displayed: $message")
        } else {
            binding.messageContainer.visibility = View.GONE
            Log.d(TAG, "No message to display")
        }
    }
    
    private fun switchToModelType(modelType: VoiceModelType) {
        currentModelType = modelType
        updateSegmentControl()
        loadModelsForType(modelType)
    }
    
    private fun updateSegmentControl() {
        // Update TTS segment
        binding.segmentTts.apply {
            isSelected = currentModelType == VoiceModelType.TTS
            setTextColor(if (isSelected) {
                ContextCompat.getColor(requireContext(), R.color.segment_control_selected_text)
            } else {
                ContextCompat.getColor(requireContext(), R.color.segment_control_normal_text)
            })
            setBackgroundResource(if (isSelected) {
                R.drawable.bg_segment_control_selected
            } else {
                R.drawable.bg_segment_control_normal
            })
        }
        
        // Update ASR segment
        binding.segmentAsr.apply {
            isSelected = currentModelType == VoiceModelType.ASR
            setTextColor(if (isSelected) {
                ContextCompat.getColor(requireContext(), R.color.segment_control_selected_text)
            } else {
                ContextCompat.getColor(requireContext(), R.color.segment_control_normal_text)
            })
            setBackgroundResource(if (isSelected) {
                R.drawable.bg_segment_control_selected
            } else {
                R.drawable.bg_segment_control_normal
            })
        }
        
        // Update title
        binding.titleText.text = getString(currentModelType.displayNameRes)
    }
    
    private fun loadModelsForType(modelType: VoiceModelType) {
        when (modelType) {
            VoiceModelType.TTS -> {
                viewModel.loadTtsModels()
            }
            VoiceModelType.ASR -> {
                viewModel.loadAsrModels()
            }
        }
    }
    
    private fun updateEmptyState(isEmpty: Boolean) {
        if (isEmpty) {
            binding.emptyStateContainer.visibility = View.VISIBLE
            binding.recyclerView.visibility = View.GONE
            binding.emptyStateText.text = when (currentModelType) {
                VoiceModelType.TTS -> getString(R.string.no_tts_models_available)
                VoiceModelType.ASR -> getString(R.string.no_asr_models_available)
            }
        } else {
            binding.emptyStateContainer.visibility = View.GONE
            binding.recyclerView.visibility = View.VISIBLE
        }
    }
    
    private fun handleModelDownloadComplete(modelId: String) {
        // Find the model in current adapter list
        val models = adapter.currentList
        val modelWrapper = models.find { it.modelMarketItem.modelId == modelId }
        
        if (modelWrapper != null && modelWrapper.downloadInfo.downloadState == DownloadState.COMPLETED) {
            Log.d(TAG, "Model download completed: $modelId")
            
            // Determine model type based on current view and set as default if needed
            when (currentModelType) {
                VoiceModelType.TTS -> {
                    val currentDefault = MainSettings.getDefaultTtsModel(requireContext())
                    if (currentDefault.isNullOrEmpty()) {
                        Log.d(TAG, "Setting $modelId as default TTS model")
                        viewModel.setDefaultTtsModel(modelId)
                        onModelChangedCallback?.invoke(VoiceModelType.TTS, modelId)
                        Toast.makeText(requireContext(), 
                            getString(R.string.default_tts_model_set, modelWrapper.modelMarketItem.modelName), 
                            Toast.LENGTH_SHORT).show()
                    }
                }
                VoiceModelType.ASR -> {
                    val currentDefault = MainSettings.getDefaultAsrModel(requireContext())
                    if (currentDefault.isNullOrEmpty()) {
                        Log.d(TAG, "Setting $modelId as default ASR model")
                        viewModel.setDefaultAsrModel(modelId)
                        onModelChangedCallback?.invoke(VoiceModelType.ASR, modelId)
                        Toast.makeText(requireContext(), 
                            getString(R.string.default_asr_model_set, modelWrapper.modelMarketItem.modelName), 
                            Toast.LENGTH_SHORT).show()
                    }
                }
            }
        }
    }

    // ModelMarketItemListener implementation
    override fun onActionClicked(item: ModelMarketItemWrapper) {
        val downloadInfo = item.downloadInfo
        when (downloadInfo.downloadState) {
            DownloadState.COMPLETED -> {
                Toast.makeText(requireContext(), getString(R.string.model_already_downloaded), Toast.LENGTH_SHORT).show()
            }
            DownloadState.NOT_START,
            DownloadState.FAILED -> {
                viewModel.startDownload(item.modelMarketItem)
            }
            DownloadState.PAUSED -> {
                viewModel.startDownload(item.modelMarketItem) // resume
            }
            DownloadState.DOWNLOADING -> {
                viewModel.pauseDownload(item.modelMarketItem)
            }
        }
    }

    override fun onDownloadOrResumeClicked(item: ModelMarketItemWrapper) {
        viewModel.startDownload(item.modelMarketItem)
    }

    override fun onPauseClicked(item: ModelMarketItemWrapper) {
        viewModel.pauseDownload(item.modelMarketItem)
    }

    override fun onDeleteClicked(item: ModelMarketItemWrapper) {
        viewModel.deleteModel(item.modelMarketItem)
    }

    override fun onUpdateClicked(item: ModelMarketItemWrapper) {
        viewModel.updateModel(item.modelMarketItem)
    }

    override fun onDefaultVoiceModelChanged(item: ModelMarketItemWrapper) {
        // Refresh adapter to update all checkbox states, ensuring only one is selected
        adapter.notifyDataSetChanged()
    }
    
    /**
     * Set callback to be executed when the default model is changed
     */
    fun setOnModelChangedCallback(callback: (VoiceModelType, String) -> Unit) {
        onModelChangedCallback = callback
    }
    
    /**
     * Set callback to be executed when the bottom sheet is dismissed
     */
    fun setOnDismissCallback(callback: () -> Unit) {
        onDismissCallback = callback
    }
    
    override fun onDismiss(dialog: DialogInterface) {
        super.onDismiss(dialog)
        onDismissCallback?.invoke()
    }
    
    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
    
    companion object {
        const val TAG = "VoiceModelMarketBottomSheet"
        private const val ARG_MESSAGE = "message"
        
        fun newInstance(message: String? = null): VoiceModelMarketBottomSheet {
            Log.d(TAG, "newInstance called with message: $message")
            val fragment = VoiceModelMarketBottomSheet()
            if (!message.isNullOrEmpty()) {
                val args = Bundle()
                args.putString(ARG_MESSAGE, message)
                fragment.arguments = args
            }
            Log.d(TAG, "newInstance created: $fragment")
            return fragment
        }
    }
} 