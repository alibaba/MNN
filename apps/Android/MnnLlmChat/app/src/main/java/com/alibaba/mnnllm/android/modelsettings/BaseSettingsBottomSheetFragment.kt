// Created by ruoyi.sjd on 2025/4/29.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.modelsettings

import android.app.Dialog
import android.os.Bundle
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

    /**
     * Load settings from config files
     */
    protected open fun loadSettings() {
        val defaultConfigFile = if (_configPath.isNullOrEmpty()) {
            ModelConfig.getDefaultConfigFile(_modelId)
        } else {
            _configPath
        }
        loadedConfig = ModelConfig.loadMergedConfig(
            defaultConfigFile!!,
            ModelConfig.getExtraConfigFile(_modelId)
        ) ?: ModelConfig.defaultConfig
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
    }

    fun setConfigPath(configPath: String?) {
        this._configPath = configPath
    }

    fun addOnSettingsDoneListener(listener: (Boolean) -> Unit) {
        onSettingsDoneListener = listener
    }
}
