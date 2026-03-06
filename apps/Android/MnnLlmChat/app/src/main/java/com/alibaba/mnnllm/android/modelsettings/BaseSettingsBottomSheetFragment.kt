// Created by ruoyi.sjd on 2025/4/29.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.modelsettings

import android.app.Dialog
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.FrameLayout
import com.alibaba.mnnllm.android.utils.BaseBottomSheetDialogFragment
import com.google.android.material.bottomsheet.BottomSheetBehavior

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
        loadSettings()
        setupUI()
        setupActionButtons()
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        outState.putString(KEY_MODEL_ID, _modelId)
        outState.putString(KEY_CONFIG_PATH, _configPath)
    }

    /**
     * Load settings from config files
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
     * Setup action buttons (Cancel, Save, Reset)
     */
    protected abstract fun setupActionButtons()

    /**
     * Save settings to config file
     */
    protected abstract fun saveSettings()

    /**
     * Reset settings to defaults
     */
    protected open fun resetSettingsToDefaults() {
        loadSettings()
    }

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
