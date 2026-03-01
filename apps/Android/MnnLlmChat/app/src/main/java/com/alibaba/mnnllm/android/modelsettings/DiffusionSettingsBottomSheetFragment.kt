// Created by ruoyi.sjd on 2025/4/29.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.modelsettings

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.core.widget.addTextChangedListener
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.databinding.FragmentDiffusionSettingsSheetBinding
import com.alibaba.mnnllm.android.mainsettings.DiffusionMemoryMode
import com.alibaba.mnnllm.android.modelsettings.ModelConfig.Companion.defaultConfig

/**
 * Settings bottom sheet fragment for Diffusion and Sana models.
 * Provides settings specific to image generation models.
 */
class DiffusionSettingsBottomSheetFragment : BaseSettingsBottomSheetFragment() {

    private var _binding: FragmentDiffusionSettingsSheetBinding? = null
    private val binding get() = _binding!!

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentDiffusionSettingsSheetBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun loadSettings() {
        super.loadSettings()
        
        // Diffusion steps
        currentConfig.diffusionSteps = currentConfig.diffusionSteps ?: defaultConfig.diffusionSteps
        binding.editDiffusionSteps.setText(currentConfig.diffusionSteps.toString())

        // Image dimensions
        currentConfig.imageWidth = currentConfig.imageWidth ?: defaultConfig.imageWidth
        binding.editImageWidth.setText(currentConfig.imageWidth.toString())

        currentConfig.imageHeight = currentConfig.imageHeight ?: defaultConfig.imageHeight
        binding.editImageHeight.setText(currentConfig.imageHeight.toString())

        // Seed
        currentConfig.diffusionSeed = currentConfig.diffusionSeed ?: defaultConfig.diffusionSeed
        binding.editDiffusionSeed.setText(currentConfig.diffusionSeed.toString())
        
        // CFG Prompt
        currentConfig.cfgPrompt = currentConfig.cfgPrompt ?: defaultConfig.cfgPrompt
        binding.editCfgPrompt.setText(currentConfig.cfgPrompt)

        // Grid size
        currentConfig.gridSize = currentConfig.gridSize ?: defaultConfig.gridSize
        binding.editGridSize.setText(currentConfig.gridSize.toString())
    }

    override fun setupUI() {
        setupDiffusionSettings()
        setupAdvancedSettings()
    }

    private fun setupDiffusionSettings() {
        binding.editDiffusionSteps.addTextChangedListener { text ->
            if (!text.isNullOrEmpty()) {
                try {
                    currentConfig.diffusionSteps = text.toString().toInt()
                } catch (e: Exception) { /* ignore */ }
            }
        }

        binding.editImageWidth.addTextChangedListener { text ->
            if (!text.isNullOrEmpty()) {
                try {
                    currentConfig.imageWidth = text.toString().toInt()
                } catch (e: Exception) { /* ignore */ }
            }
        }

        binding.editImageHeight.addTextChangedListener { text ->
            if (!text.isNullOrEmpty()) {
                try {
                    currentConfig.imageHeight = text.toString().toInt()
                } catch (e: Exception) { /* ignore */ }
            }
        }

        binding.editDiffusionSeed.addTextChangedListener { text ->
            if (!text.isNullOrEmpty()) {
                try {
                    currentConfig.diffusionSeed = text.toString().toLong()
                } catch (e: Exception) { /* ignore */ }
            }
        }

        binding.editCfgPrompt.addTextChangedListener { text ->
            if (!text.isNullOrEmpty()) {
                currentConfig.cfgPrompt = text.toString()
            }
        }

        binding.editGridSize.addTextChangedListener { text ->
            if (!text.isNullOrEmpty()) {
                try {
                    currentConfig.gridSize = text.toString().toInt()
                } catch (e: Exception) { /* ignore */ }
            }
        }
    }

    private fun setupAdvancedSettings() {
        // Diffusion memory mode
        val memoryModeValue = currentConfig.diffusionMemoryMode 
            ?: defaultConfig.diffusionMemoryMode 
            ?: DiffusionMemoryMode.MEMORY_MODE_SAVING.value
        val memoryModeEntries = DiffusionMemoryMode.values().toList()

        fun getMemoryModeString(mode: DiffusionMemoryMode): String {
            return when(mode) {
                DiffusionMemoryMode.MEMORY_MODE_SAVING -> getString(R.string.diffusion_mode_memory_saving)
                DiffusionMemoryMode.MEMORY_MODE_ENOUGH -> getString(R.string.diffusion_mode_memory_enough)
                DiffusionMemoryMode.MEMORY_MODE_BALANCE -> getString(R.string.diffusion_mode_memory_balance)
            }
        }

        val currentMemoryMode = memoryModeEntries.find { it.value == memoryModeValue } 
            ?: DiffusionMemoryMode.MEMORY_MODE_SAVING

        binding.dropdownDiffusionMemoryMode.setCurrentItem(currentMemoryMode)
        binding.dropdownDiffusionMemoryMode.setDropDownItems(
            memoryModeEntries,
            itemToString = { getMemoryModeString(it as DiffusionMemoryMode) },
            onDropdownItemSelected = { _, item ->
                currentConfig.diffusionMemoryMode = (item as DiffusionMemoryMode).value
            }
        )

        // Backend
        val backendOptions = listOf("cpu", "opencl")
        val currentBackend = currentConfig.backendType.takeIf { it in backendOptions } ?: "opencl"
        binding.dropdownBackend.setCurrentItem(currentBackend)
        binding.dropdownBackend.setDropDownItems(
            backendOptions,
            itemToString = { it.toString() },
            onDropdownItemSelected = { _, item ->
                currentConfig.backendType = item.toString()
            }
        )
    }

    override fun setupActionButtons() {
        binding.buttonCancel.setOnClickListener {
            dismiss()
        }
        binding.buttonSave.setOnClickListener {
            saveSettings()
            dismiss()
        }
        binding.buttonReset.setOnClickListener {
            resetSettingsToDefaults()
        }
    }

    override fun saveSettings() {
        var needRecreate = needRecreateActivity
        var needSaveConfig = false

        if (currentConfig == loadedConfig) {
            return
        }
        
        // Check what changed
        if (currentConfig.diffusionMemoryMode != loadedConfig.diffusionMemoryMode) {
            needSaveConfig = true
            needRecreate = true
        } else if (currentConfig.backendType != loadedConfig.backendType) {
            needSaveConfig = true
            needRecreate = true
        } else if (currentConfig.diffusionSteps != loadedConfig.diffusionSteps ||
                   currentConfig.imageWidth != loadedConfig.imageWidth ||
                   currentConfig.imageHeight != loadedConfig.imageHeight ||
                   currentConfig.diffusionSeed != loadedConfig.diffusionSeed ||
                   currentConfig.cfgPrompt != loadedConfig.cfgPrompt ||
                   currentConfig.gridSize != loadedConfig.gridSize) {
            needSaveConfig = true
            needRecreate = false
        }

        if (needSaveConfig) {
            ModelConfig.saveConfig(ModelConfig.getExtraConfigFile(modelId), currentConfig)
        }
        onSettingsDoneListener?.invoke(needRecreate)
    }

    override fun resetSettingsToDefaults() {
        super.resetSettingsToDefaults()
        // Re-populate UI with loaded values
        binding.editDiffusionSteps.setText(currentConfig.diffusionSteps.toString())
        binding.editImageWidth.setText(currentConfig.imageWidth.toString())
        binding.editImageHeight.setText(currentConfig.imageHeight.toString())
        binding.editDiffusionSeed.setText(currentConfig.diffusionSeed.toString())
        binding.editCfgPrompt.setText(currentConfig.cfgPrompt)
        binding.editGridSize.setText(currentConfig.gridSize.toString())
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }

    companion object {
        const val TAG = "DiffusionSettingsBottomSheetFragment"
    }
}
