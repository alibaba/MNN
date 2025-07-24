package com.alibaba.mnnllm.android.modelmarket

import android.content.Context
import android.widget.CheckBox
import android.widget.Toast
import androidx.core.content.ContextCompat
import com.alibaba.mls.api.download.DownloadInfo
import com.alibaba.mls.api.download.DownloadState
import com.alibaba.mls.api.download.ModelDownloadManager
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.mainsettings.MainSettings
import com.alibaba.mnnllm.android.model.ModelUtils
import com.alibaba.mnnllm.android.utils.FileUtils
import com.google.android.material.button.MaterialButton

/**
 * Delegate class to handle voice model (TTS/ASR) specific logic in MarketItemHolder
 */
class MarketHolderVoiceDelegate(
    private val context: Context,
    private val modelDownloadManager: ModelDownloadManager
) {
    
    // Holds the CheckBox for voice model
    private var checkboxVoiceModel: CheckBox? = null
    
    // Callback to notify when a voice model is set as default
    private var onVoiceModelChangedCallback: ((VoiceModelType, String) -> Unit)? = null
    
    enum class VoiceModelType(val displayName: String) {
        TTS("TTS"),
        ASR("ASR"),
        NONE("NONE")
    }
    
    /**
     * Set callback to be executed when a voice model is set as default
     */
    fun setOnVoiceModelChangedCallback(callback: (VoiceModelType, String) -> Unit) {
        onVoiceModelChangedCallback = callback
    }
    
    /**
     * Attach a CheckBox for voiceDelegate to manage
     */
    fun attachCheckbox(checkbox: CheckBox) {
        this.checkboxVoiceModel = checkbox
    }
    
    /**
     * Hide the checkbox, used for non-voice models
     */
    fun hideCheckbox() {
        checkboxVoiceModel?.visibility = android.view.View.GONE
        checkboxVoiceModel?.setOnCheckedChangeListener(null)
    }
    
    /**
     * Determine if the model is a voice model and return its type
     */
    fun getVoiceModelType(modelMarketItem: ModelMarketItem): VoiceModelType {
        return when {
            ModelUtils.isTtsModelByTags(modelMarketItem.tags) -> VoiceModelType.TTS
            ModelUtils.isAsrModelByTags(modelMarketItem.tags) -> VoiceModelType.ASR
            else -> VoiceModelType.NONE
        }
    }
    
    /**
     * Check if the model is currently set as default for its type
     */
    fun isDefaultModel(modelId: String, type: VoiceModelType): Boolean {
        return when (type) {
            VoiceModelType.TTS -> MainSettings.isDefaultTtsModel(context, modelId)
            VoiceModelType.ASR -> MainSettings.isDefaultAsrModel(context, modelId)
            VoiceModelType.NONE -> false
        }
    }
    
    /**
     * Set voice model UI components visibility and state
     */
    fun setVoiceModelUI(
        button: MaterialButton,
        checkbox: CheckBox,
        isCompleted: Boolean,
        isDefault: Boolean,
        type: VoiceModelType,
        modelMarketItem: ModelMarketItem,
        onModelUpdated: () -> Unit
    ) {
        // Record the checkbox
        attachCheckbox(checkbox)
        if (isCompleted) {
            // Hide button and show checkbox for completed voice models
            button.visibility = android.view.View.GONE
            checkbox.visibility = android.view.View.VISIBLE
            checkbox.text = ""

            // Set checkbox state without triggering listener
            checkbox.setOnCheckedChangeListener(null)
            checkbox.isChecked = isDefault
            
            // Set up checkbox change listener
            setupCheckboxListener(checkbox, modelMarketItem, type, onModelUpdated)
        } else {
            // Show button and hide checkbox for non-completed voice models
            button.visibility = android.view.View.VISIBLE
            checkbox.visibility = android.view.View.GONE
            
            // Clear any existing listener
            checkbox.setOnCheckedChangeListener(null)
        }
    }
    
    /**
     * Set up checkbox change listener for voice models
     */
    private fun setupCheckboxListener(
        checkbox: CheckBox,
        modelMarketItem: ModelMarketItem,
        type: VoiceModelType,
        onModelUpdated: () -> Unit
    ) {
        checkbox.setOnCheckedChangeListener { _, isChecked ->
            handleVoiceModelCheckboxChange(modelMarketItem, type, isChecked, checkbox, onModelUpdated)
        }
    }
    
    /**
     * Update status text for voice models
     */
    fun getVoiceModelStatusText(
        downloadInfo: DownloadInfo,
        modelId: String,
        isDefault: Boolean,
        type: VoiceModelType
    ): String {
        return FileUtils.getFileSizeString(
            modelDownloadManager.getDownloadedFile(modelId)
        )
    }
    
    /**
     * Handle voice model checkbox change
     */
    fun handleVoiceModelCheckboxChange(
        modelMarketItem: ModelMarketItem,
        type: VoiceModelType,
        isChecked: Boolean,
        checkbox: CheckBox,
        onModelUpdated: () -> Unit
    ) {
        if (isChecked) {
            // Set as default model
            when (type) {
                VoiceModelType.TTS -> {
                    MainSettings.setDefaultTtsModel(context, modelMarketItem.modelId)
                    onVoiceModelChangedCallback?.invoke(VoiceModelType.TTS, modelMarketItem.modelId)
                    Toast.makeText(
                        context,
                        context.getString(R.string.default_tts_model_set, modelMarketItem.modelName),
                        Toast.LENGTH_SHORT
                    ).show()
                }
                VoiceModelType.ASR -> {
                    MainSettings.setDefaultAsrModel(context, modelMarketItem.modelId)
                    onVoiceModelChangedCallback?.invoke(VoiceModelType.ASR, modelMarketItem.modelId)
                    Toast.makeText(
                        context,
                        context.getString(R.string.default_asr_model_set, modelMarketItem.modelName),
                        Toast.LENGTH_SHORT
                    ).show()
                }
                VoiceModelType.NONE -> {
                    // Should not happen, but handle gracefully
                    return
                }
            }
            
            // Notify that the model has been updated
            onModelUpdated()
        } else {
            // Uncheck is not allowed, keep it checked
            // This prevents users from unchecking without selecting another default
            checkbox.setOnCheckedChangeListener(null)
            checkbox.isChecked = true
            // Re-setup the listener
            setupCheckboxListener(checkbox, modelMarketItem, type, onModelUpdated)
            
            Toast.makeText(
                context,
                context.getString(R.string.cannot_uncheck_default_model),
                Toast.LENGTH_SHORT
            ).show()
        }
    }
    
    /**
     * Check if a download state and model combination should be handled by voice delegate
     */
    fun shouldHandleModel(downloadState: Int, modelMarketItem: ModelMarketItem): Boolean {
        return downloadState == DownloadState.COMPLETED && 
               getVoiceModelType(modelMarketItem) != VoiceModelType.NONE
    }
} 