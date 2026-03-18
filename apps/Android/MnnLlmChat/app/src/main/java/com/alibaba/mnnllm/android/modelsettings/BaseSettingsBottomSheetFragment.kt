// Created by ruoyi.sjd on 2025/4/29.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.modelsettings

import android.app.Dialog
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.FrameLayout
import androidx.lifecycle.lifecycleScope
import com.alibaba.mnnllm.android.utils.BaseBottomSheetDialogFragment
import com.google.android.material.bottomsheet.BottomSheetBehavior
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * Base class for model settings bottom sheet fragments.
 * Provides common functionality for loading/saving config and UI setup.
 */
abstract class BaseSettingsBottomSheetFragment : BaseBottomSheetDialogFragment() {

    protected lateinit var loadedConfig: ModelConfig
    private var _modelId: String = ""
    protected val modelId: String get() = _modelId
    protected lateinit var currentConfig: ModelConfig
    private var _configPath: String? = null
    protected val configPath: String? get() = _configPath
    protected var needRecreateActivity = false
    protected var onSettingsDoneListener: ((Boolean) -> Unit)? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val savedModelId = savedInstanceState?.getString(KEY_MODEL_ID)
        val argModelId = arguments?.getString(KEY_MODEL_ID)
        _modelId = savedModelId ?: argModelId ?: _modelId

        _configPath = if (savedInstanceState?.containsKey(KEY_CONFIG_PATH) == true) {
            savedInstanceState.getString(KEY_CONFIG_PATH)
        } else {
            arguments?.getString(KEY_CONFIG_PATH) ?: _configPath
        }
    }

    override fun onStart() {
        super.onStart()
        val dialog: Dialog? = dialog
        if (dialog != null) {
            val bottomSheet: FrameLayout? = dialog.findViewById(com.google.android.material.R.id.design_bottom_sheet)
            if (bottomSheet != null) {
                val behavior = BottomSheetBehavior.from(bottomSheet)
                bottomSheet.post {
                    behavior.state = BottomSheetBehavior.STATE_EXPANDED
                }
                behavior.skipCollapsed = false
            }
        }
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        // Load config off main thread to avoid ANR (file I/O)
        lifecycleScope.launch {
            loadSettingsAsync()
            setupUI()
            refreshUIFromConfig()
            setupActionButtons()
        }
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        outState.putString(KEY_MODEL_ID, _modelId)
        outState.putString(KEY_CONFIG_PATH, _configPath)
    }

    /**
     * Load settings from config files (runs on IO dispatcher, then updates on Main)
     */
    private suspend fun loadSettingsAsync() {
        val config = withContext(Dispatchers.IO) {
            val defaultConfigFile = resolveConfigFilePath(_modelId, _configPath, ModelConfig::getDefaultConfigFile)
            if (defaultConfigFile.isNullOrBlank()) {
                Log.w(TAG, "Missing config path for modelId=$_modelId, fallback to default config")
                ModelConfig.defaultConfig.deepCopy()
            } else {
                ModelConfig.loadMergedConfig(
                    defaultConfigFile,
                    ModelConfig.getExtraConfigFile(_modelId)
                ) ?: ModelConfig.defaultConfig.deepCopy()
            }
        }
        loadedConfig = config
        currentConfig = loadedConfig.deepCopy()
    }

    /**
     * Load settings from config files (synchronous, for internal use from coroutine)
     */
    protected open fun loadSettings() {
        val defaultConfigFile = resolveConfigFilePath(_modelId, _configPath, ModelConfig::getDefaultConfigFile)
        loadedConfig = if (defaultConfigFile.isNullOrBlank()) {
            Log.w(TAG, "Missing config path for modelId=$_modelId, fallback to default config")
            ModelConfig.defaultConfig.deepCopy()
        } else {
            ModelConfig.loadMergedConfig(
                defaultConfigFile,
                ModelConfig.getExtraConfigFile(_modelId)
            ) ?: ModelConfig.defaultConfig.deepCopy()
        }
        currentConfig = loadedConfig.deepCopy()
    }

    /**
     * Setup UI components - to be implemented by subclasses
     */
    protected abstract fun setupUI()

    /**
     * Refresh UI from currentConfig after load. Override to populate EditTexts etc.
     * Called after loadSettingsAsync and setupUI so saved values (e.g. system prompt) are displayed.
     */
    protected open fun refreshUIFromConfig() {}

    /**
     * Setup action buttons (Cancel, Save, Reset)
     */
    protected abstract fun setupActionButtons()

    /**
     * Save settings to config file
     */
    protected abstract fun saveSettings()

    /**
     * Reset settings to defaults. Deletes custom_config.json so base config.json is used,
     * then reloads. Ensures default system prompt and other defaults are restored.
     */
    protected open fun resetSettingsToDefaults() {
        lifecycleScope.launch {
            withContext(Dispatchers.IO) {
                ModelConfig.deleteExtraConfig(_modelId)
            }
            loadSettingsAsync()
            onAfterSettingsReset()
        }
    }

    /**
     * Called after settings are reset. Override in subclasses to update UI.
     */
    protected open fun onAfterSettingsReset() {}

    fun setModelId(modelId: String) {
        this._modelId = modelId
        if (!isAdded) {
            val args = (arguments ?: Bundle()).apply {
                putString(KEY_MODEL_ID, modelId)
            }
            arguments = args
        }
    }

    fun setConfigPath(configPath: String?) {
        this._configPath = configPath
        if (!isAdded) {
            val args = (arguments ?: Bundle()).apply {
                putString(KEY_CONFIG_PATH, configPath)
            }
            arguments = args
        }
    }

    fun addOnSettingsDoneListener(listener: (Boolean) -> Unit) {
        onSettingsDoneListener = listener
    }

    companion object {
        private const val TAG = "BaseSettingsBottomSheet"
        private const val KEY_MODEL_ID = "settings_model_id"
        private const val KEY_CONFIG_PATH = "settings_config_path"

        internal fun resolveConfigFilePath(
            modelId: String,
            configPath: String?,
            defaultConfigProvider: (String) -> String?
        ): String? {
            if (!configPath.isNullOrBlank()) {
                return configPath
            }
            if (modelId.isBlank()) {
                return null
            }
            return defaultConfigProvider(modelId)
        }
    }
}
