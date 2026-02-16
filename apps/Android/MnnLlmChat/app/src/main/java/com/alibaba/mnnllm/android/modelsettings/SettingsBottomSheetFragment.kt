// Created by ruoyi.sjd on 2025/4/29.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.modelsettings

import android.annotation.SuppressLint
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.SeekBar
import android.widget.Toast
import androidx.core.view.isVisible
import androidx.core.widget.addTextChangedListener
import com.alibaba.mls.api.ModelItem
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.databinding.FragmentSettingsSheetBinding
import com.alibaba.mnnllm.android.databinding.SettingsRowSliderSwitchBinding
import com.alibaba.mnnllm.android.llm.LlmSession
import com.alibaba.mnnllm.android.modelsettings.ModelConfig.Companion.defaultConfig
import com.alibaba.mnnllm.android.utils.MmapUtils
import java.util.Locale

enum class SamplerType(val value: String) {
    Greedy("greedy"),
    Temperature("temperature"),
    TopK("topK"),
    TopP("topP"),
    MinP("minP"),
    Tfs("tfsZ"),
    Typical("typical"),
    Penalty("penalty"),
    Mixed("mixed")}

val mainSamplerTypes = listOf (
    SamplerType.Greedy,
    SamplerType.Penalty,
    SamplerType.Mixed
)

/**
 * Settings bottom sheet fragment for LLM models.
 * Provides settings specific to language models like sampler settings, max tokens, system prompt, etc.
 */
class SettingsBottomSheetFragment : BaseSettingsBottomSheetFragment() {

    private var modelItem: ModelItem? = null
    private var chatSession: LlmSession? = null
    private var _binding: FragmentSettingsSheetBinding? = null
    private val binding get() = _binding!!
    private var currentSamplerType: SamplerType = SamplerType.Mixed
    private var penaltySamplerValue: String = "greedy"

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentSettingsSheetBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun loadSettings() {
        super.loadSettings()
        updateSamplerSettings()
        
        // Max tokens
        currentConfig.maxNewTokens = currentConfig.maxNewTokens ?: defaultConfig.maxNewTokens
        binding.editMaxNewTokens.setText(currentConfig.maxNewTokens.toString())

        // System prompt
        currentConfig.systemPrompt = currentConfig.systemPrompt ?: defaultConfig.systemPrompt
        binding.editTextSystemPrompt.setText(currentConfig.systemPrompt)
    }

    override fun setupUI() {
        setupModelConfig()
        setupSamplerSettings()
        setupAdvancedConfigs()
        setupMaxTokenListener()
        setupSystemPromptListener()
        hideDiffusionSettings()
    }

    private fun hideDiffusionSettings() {
        // Hide all diffusion-related settings for LLM models
        binding.dropdownDiffusionMemoryMode.isVisible = false
        binding.idLabelDiffusionSteps.isVisible = false
        binding.inputLayoutDiffusionSteps.isVisible = false
        binding.idLabelImageWidth.isVisible = false
        binding.inputLayoutImageWidth.isVisible = false
        binding.idLabelImageHeight.isVisible = false
        binding.inputLayoutImageHeight.isVisible = false
        binding.idLabelDiffusionSeed.isVisible = false
        binding.inputLayoutDiffusionSeed.isVisible = false
        binding.idLabelCfgPrompt.isVisible = false
        binding.inputLayoutCfgPrompt.isVisible = false
        binding.idLabelGridSize.isVisible = false
        binding.inputLayoutGridSize.isVisible = false
    }

    @SuppressLint("SetTextI18n")
    private fun setupAdvancedConfigs() {
        // Mmap settings
        binding.mmapSettingsItem.isChecked = currentConfig.useMmap ?: defaultConfig.useMmap!!
        binding.mmapSettingsItem.setOnCheckedChangeListener { _, isChecked ->
            currentConfig.useMmap = isChecked
        }
        binding.buttonClearMmapCache.setOnClickListener {
            val success = MmapUtils.clearMmapCache(modelId)
            if (success) {
                needRecreateActivity = true
                Toast.makeText(requireActivity(), R.string.mmap_cacche_cleared, Toast.LENGTH_LONG).show()
            } else {
                Toast.makeText(requireActivity(), R.string.mmap_not_used, Toast.LENGTH_LONG).show()
            }
        }
        
        // Precision
        binding.dropdownPrecision.setCurrentItem(currentConfig.precision ?: defaultConfig.precision!!)
        binding.dropdownPrecision.setDropDownItems(
            listOf("low", "high"),
            itemToString = { it.toString() },
            onDropdownItemSelected = { _, item ->
                currentConfig.precision = item.toString()
            },
        )

        // Thread num
        val threadNum = currentConfig.threadNum ?: defaultConfig.threadNum!!
        binding.etThreadNum.setText(threadNum.toString())
        binding.etThreadNum.addTextChangedListener { text ->
            if (text.isNullOrEmpty()) {
                return@addTextChangedListener
            }
            currentConfig.threadNum = text.toString().toInt()
        }

        // Backend
        val backendOptions = listOf("cpu", "opencl")
        val currentBackend = currentConfig.backendType.takeIf { it in backendOptions } ?: "cpu"
        binding.dropdownBackend.setCurrentItem(currentBackend)
        binding.dropdownBackend.setDropDownItems(
            backendOptions,
            itemToString = { it.toString() },
            onDropdownItemSelected = { _, item ->
                currentConfig.backendType = item.toString()
            },
        )
    }

    private fun setupMaxTokenListener() {
        binding.editMaxNewTokens.addTextChangedListener { text ->
            if (text.isNullOrEmpty()) {
                return@addTextChangedListener
            }
            currentConfig.maxNewTokens = text.toString().toInt()
        }
    }

    private fun setupSystemPromptListener() {
        binding.editTextSystemPrompt.addTextChangedListener { text ->
            if (text.isNullOrEmpty()) {
                return@addTextChangedListener
            }
            currentConfig.systemPrompt = text.toString()
        }
    }

    private fun setupSamplerSettings() {
        updateSamplerSettings()
    }

    private fun updateSamplerSettings() {
        if (currentConfig.samplerType == SamplerType.Mixed.value) {
            currentSamplerType = SamplerType.Mixed
        } else if (currentConfig.samplerType == SamplerType.Penalty.value) {
            currentSamplerType = SamplerType.Penalty
        } else {
            currentSamplerType = SamplerType.Greedy
        }
        binding.dropdownSamplerType.setDropDownItems(
            mainSamplerTypes,
            itemToString = {
                samplerTypeToString(it as SamplerType)
            },
            onDropdownItemSelected = { _, item ->
                currentSamplerType = item as SamplerType
                currentConfig.samplerType = item.value
                updateSamplerSettings()
            },
        )
        binding.dropdownSamplerType.setCurrentItem(currentSamplerType)
        updateSamplerSettingsVisibility()
        
        if (currentSamplerType == SamplerType.Mixed) {
            setupMixedSettings()
        } else if (currentSamplerType == SamplerType.Penalty) {
            setupPenaltySettings()
        }
    }


    private fun toggleEnable(items: MutableList<String>, item: SamplerType, enabled: Boolean) {
        if (enabled) {
            if (!items.contains(item.value)) {
                items.add(item.value)
            }
        } else if (items.contains(item.value)) {
            items.remove(item.value)
        }
    }

    private fun setupMixedSettings() {
        if (currentConfig.mixedSamplers == null) {
            currentConfig.mixedSamplers = mutableListOf(
                SamplerType.TopK.value,
                SamplerType.TopP.value,
                SamplerType.MinP.value,
                SamplerType.Temperature.value,
                SamplerType.Penalty.value
            )
        }
        currentConfig.topK = currentConfig.topK ?: defaultConfig.topK!!
        setupSliderSwitchRow(
            rowBinding = SettingsRowSliderSwitchBinding.bind(binding.rowMixedTopK.root),
            label = samplerTypeToString(SamplerType.TopK),
            initialValue = (currentConfig.topK!!).toFloat(),
            initialEnabled = currentConfig.mixedSamplers!!.contains(SamplerType.TopK.value),
            valueRange = 1f..100f,
            decimalPlaces = 0,
            onValueChange = { currentConfig.topK = it.toInt() },
            onEnabledChange = { toggleEnable(currentConfig.mixedSamplers!!, SamplerType.TopK, it)}
        )
        currentConfig.tfsZ = currentConfig.tfsZ ?: defaultConfig.tfsZ!!
        setupSliderSwitchRow(
            rowBinding = SettingsRowSliderSwitchBinding.bind(binding.rowMixedTfsZ.root),
            label = samplerTypeToString(SamplerType.Tfs),
            initialValue = (currentConfig.tfsZ!!),
            initialEnabled = currentConfig.mixedSamplers!!.contains(SamplerType.Tfs.value),
            valueRange = 0f..1f,
            decimalPlaces = 0,
            onValueChange = { currentConfig.tfsZ = it },
            onEnabledChange = { toggleEnable(currentConfig.mixedSamplers!!, SamplerType.Tfs, it)}
        )
        currentConfig.typical = currentConfig.typical ?: defaultConfig.typical!!
        setupSliderSwitchRow(
            rowBinding = SettingsRowSliderSwitchBinding.bind(binding.rowMixedTypical.root),
            label = samplerTypeToString(SamplerType.Typical),
            initialValue = (currentConfig.typical!!).toFloat(),
            initialEnabled = currentConfig.mixedSamplers!!.contains(SamplerType.Typical.value),
            valueRange = 0f..1f,
            decimalPlaces = 0,
            onValueChange = { currentConfig.typical = it },
            onEnabledChange = { toggleEnable(currentConfig.mixedSamplers!!, SamplerType.Typical, it)}
        )
        currentConfig.topP = currentConfig.topP ?: defaultConfig.topP!!
        setupSliderSwitchRow(
            rowBinding = SettingsRowSliderSwitchBinding.bind(binding.rowMixedTopP.root),
            label = samplerTypeToString(SamplerType.TopP),
            initialValue = (currentConfig.topP!!).toFloat(),
            initialEnabled = currentConfig.mixedSamplers!!.contains(SamplerType.TopP.value),
            valueRange = 0f..1f,
            decimalPlaces = 2,
            onValueChange = { currentConfig.topP = it },
            onEnabledChange = { toggleEnable(currentConfig.mixedSamplers!!, SamplerType.TopP, it)}
        )
        currentConfig.minP = currentConfig.minP ?: defaultConfig.minP!!
        setupSliderSwitchRow(
            rowBinding = SettingsRowSliderSwitchBinding.bind(binding.rowMixedMinP.root),
            label = samplerTypeToString(SamplerType.MinP),
            initialValue = (currentConfig.minP!!).toFloat(),
            initialEnabled = currentConfig.mixedSamplers!!.contains(SamplerType.MinP.value),
            valueRange = 0f..1f,
            decimalPlaces = 2,
            onValueChange = { currentConfig.minP = it },
            onEnabledChange = { toggleEnable(currentConfig.mixedSamplers!!, SamplerType.MinP, it)}
        )
        currentConfig.temperature = currentConfig.temperature ?: defaultConfig.temperature!!
        setupSliderSwitchRow(
            rowBinding = SettingsRowSliderSwitchBinding.bind(binding.rowMixedTemp.root),
            label = samplerTypeToString(SamplerType.Temperature),
            initialValue = (currentConfig.temperature!!).toFloat(),
            initialEnabled = currentConfig.mixedSamplers!!.contains(SamplerType.Temperature.value),
            valueRange = 0f..2f,
            decimalPlaces = 2,
            onValueChange = { currentConfig.temperature = it },
            onEnabledChange = { toggleEnable(currentConfig.mixedSamplers!!, SamplerType.Temperature, it)}
        )

        // Add penalty settings to mixed sampler
        setupMixedPenaltySettings()
    }

    private fun setupMixedPenaltySettings() {
        val isPenaltyEnabled = currentConfig.mixedSamplers!!.contains(SamplerType.Penalty.value)
        
        currentConfig.penalty = currentConfig.penalty ?: defaultConfig.penalty!!
        setupSliderSwitchRow(
            rowBinding = SettingsRowSliderSwitchBinding.bind(binding.rowMixedPenalty.root),
            label = "Penalty",
            initialValue = currentConfig.penalty!!,
            initialEnabled = isPenaltyEnabled,
            valueRange = 0f..5f,
            decimalPlaces = 2,
            onValueChange = { currentConfig.penalty = it },
            onEnabledChange = { enabled ->
                toggleEnable(currentConfig.mixedSamplers!!, SamplerType.Penalty, enabled)
                updateMixedPenaltyControlsState(enabled)
            }
        )

        currentConfig.nGram = currentConfig.nGram ?: defaultConfig.nGram!!
        setupSliderSwitchRow(
            rowBinding = SettingsRowSliderSwitchBinding.bind(binding.rowMixedNgramSize.root),
            label = "N-gram Size",
            initialValue = (currentConfig.nGram!!).toFloat(),
            initialEnabled = isPenaltyEnabled,
            valueRange = 1f..16f,
            decimalPlaces = 0,
            onValueChange = { currentConfig.nGram = it.toInt() },
            switchVisible = false
        )

        currentConfig.nGramFactor = currentConfig.nGramFactor ?: defaultConfig.nGramFactor!!
        setupSliderSwitchRow(
            rowBinding = SettingsRowSliderSwitchBinding.bind(binding.rowMixedNgramFactor.root),
            label = "N-gram Factor",
            initialValue = currentConfig.nGramFactor!!,
            initialEnabled = isPenaltyEnabled,
            valueRange = 1f..2f,
            decimalPlaces = 1,
            onValueChange = { currentConfig.nGramFactor = it },
            switchVisible = false
        )

        penaltySamplerValue = currentConfig.penaltySampler ?: defaultConfig.penaltySampler!!
        binding.dropdownMixedPenaltySampler.setDropDownItems(listOf("greedy", "temperature")) { _, value ->
            penaltySamplerValue = value.toString()
            currentConfig.penaltySampler = penaltySamplerValue
        }
        binding.dropdownMixedPenaltySampler.setCurrentItem(penaltySamplerValue)
        
        // Initialize the controls state based on penalty switch
        updateMixedPenaltyControlsState(isPenaltyEnabled)
    }

    private fun updateMixedPenaltyControlsState(enabled: Boolean) {
        // Update N-gram Size control
        val ngramSizeBinding = SettingsRowSliderSwitchBinding.bind(binding.rowMixedNgramSize.root)
        ngramSizeBinding.seekbar.isEnabled = enabled
        ngramSizeBinding.labelSlider.alpha = if (enabled) 1.0f else 0.5f
        ngramSizeBinding.valueSlider.alpha = if (enabled) 1.0f else 0.5f
        
        // Update N-gram Factor control
        val ngramFactorBinding = SettingsRowSliderSwitchBinding.bind(binding.rowMixedNgramFactor.root)
        ngramFactorBinding.seekbar.isEnabled = enabled
        ngramFactorBinding.labelSlider.alpha = if (enabled) 1.0f else 0.5f
        ngramFactorBinding.valueSlider.alpha = if (enabled) 1.0f else 0.5f
        
        // Update dropdown control
        binding.dropdownMixedPenaltySampler.isEnabled = enabled
        binding.dropdownMixedPenaltySampler.alpha = if (enabled) 1.0f else 0.5f
    }

    private fun setupPenaltySettings() {
        setupSliderSwitchRow(
            rowBinding = SettingsRowSliderSwitchBinding.bind(binding.rowPenaltyPenalty.root),
            label = "Penalty",
            initialValue = currentConfig.penalty ?: defaultConfig.penalty!!,
            initialEnabled = true,
            valueRange = 0f..5f,
            decimalPlaces = 2,
            onValueChange = { currentConfig.penalty = it },
            switchVisible = false
        )

        setupSliderSwitchRow(
            rowBinding = SettingsRowSliderSwitchBinding.bind(binding.rowPenaltyNgramSize.root),
            label = "N-gram Size",
            initialValue = (currentConfig.nGram ?: defaultConfig.nGram!!).toFloat(),
            initialEnabled = true,
            valueRange = 1f..16f,
            decimalPlaces = 0,
            onValueChange = { currentConfig.nGram = it.toInt() },
            switchVisible = false
        )

        setupSliderSwitchRow(
            rowBinding = SettingsRowSliderSwitchBinding.bind(binding.rowPenaltyNgramFactor.root),
            label = "N-gram Factor",
            initialValue = currentConfig.nGramFactor ?: defaultConfig.nGramFactor!!,
            initialEnabled = true,
            valueRange = 1f..2f,
            decimalPlaces = 1,
            onValueChange = { currentConfig.nGramFactor = it },
            switchVisible = false
        )
        penaltySamplerValue = currentConfig.penaltySampler ?: defaultConfig.penaltySampler!!
        binding.dropdownPenaltySampler.setDropDownItems(listOf("greedy", "temperature")) { _, value ->
            penaltySamplerValue = value.toString()
            currentConfig.penaltySampler = penaltySamplerValue
        }
        binding.dropdownPenaltySampler.setCurrentItem(penaltySamplerValue)
    }

    private fun setupSliderSwitchRow(
        rowBinding: SettingsRowSliderSwitchBinding,
        label: String,
        initialValue: Float,
        initialEnabled: Boolean,
        valueRange: ClosedFloatingPointRange<Float>,
        decimalPlaces: Int,
        onValueChange: (Float) -> Unit,
        onEnabledChange: (Boolean) -> Unit = {},
        switchVisible: Boolean = true
    ) {
        val valueFormat = "%.${decimalPlaces}f"
        rowBinding.labelSlider.text = label
        rowBinding.valueSlider.text = String.format(Locale.US, valueFormat, initialValue)

        val maxProgress = 1000
        val range = valueRange.endInclusive - valueRange.start
        rowBinding.seekbar.max = maxProgress
        rowBinding.seekbar.progress = ((initialValue - valueRange.start) / range * maxProgress).toInt()

        rowBinding.seekbar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                if (fromUser) {
                    val newValue = valueRange.start + (progress.toFloat() / maxProgress) * range
                    val clampedValue = newValue.coerceIn(valueRange)
                    rowBinding.valueSlider.text = String.format(Locale.US, valueFormat, clampedValue)
                    onValueChange(clampedValue)
                }
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })
        rowBinding.switchSlider.isChecked = initialEnabled
        rowBinding.seekbar.isEnabled = initialEnabled
        if (switchVisible) {
            rowBinding.switchSlider.setOnCheckedChangeListener { _, isChecked ->
                rowBinding.seekbar.isEnabled = isChecked
                onEnabledChange(isChecked)
            }
        } else {
            rowBinding.switchSlider.visibility = View.GONE
        }
    }

    private fun setupModelConfig() {
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

    private fun samplerTypeToString(type: SamplerType): String {
        return when (type) {
            SamplerType.Mixed -> "Mixed"
            SamplerType.Penalty -> "Penalty"
            SamplerType.TopP -> "Top P"
            SamplerType.Greedy -> "Greedy"
            SamplerType.Temperature -> "Temperature"
            SamplerType.TopK -> "Top K"
            SamplerType.MinP -> "Min P"
            SamplerType.Tfs -> "TFS-Z"
            SamplerType.Typical -> "Typical"
        }
    }

    private fun updateSamplerSettingsVisibility() {
        binding.containerMixedSettings.isVisible = (currentSamplerType == SamplerType.Mixed)
        binding.containerPenaltySettings.isVisible = (currentSamplerType == SamplerType.Penalty)
        binding.containerTopPSettings.isVisible = (currentSamplerType == SamplerType.TopP)
    }

    override fun saveSettings() {
        var needRecreate = this.needRecreateActivity
        var needSaveConfig = false
        if (currentConfig == loadedConfig) {
            return
        } else if (!currentConfig.samplerEquals(loadedConfig)) {
            needSaveConfig = true
            needRecreate = true
        } else if (currentConfig.maxNewTokens != loadedConfig.maxNewTokens) {
            needSaveConfig = true
            chatSession?.updateMaxNewTokens(currentConfig.maxNewTokens!!)
            needRecreate = false
        } else if (currentConfig.systemPrompt != loadedConfig.systemPrompt) {
            needSaveConfig = true
            chatSession?.updateSystemPrompt(currentConfig.systemPrompt!!)
            needRecreate = false
        } else if (currentConfig.useMmap != loadedConfig.useMmap) {
            needSaveConfig = true
            needRecreate = true
        } else if (currentConfig.precision != loadedConfig.precision) {
            needSaveConfig = true
            needRecreate = true
        } else if (currentConfig.threadNum != loadedConfig.threadNum) {
            needSaveConfig = true
            needRecreate = true
        } else if (currentConfig.backendType != loadedConfig.backendType) {
            needSaveConfig = true
            needRecreate = true
        }
        if (needSaveConfig) {
            ModelConfig.saveConfig(ModelConfig.getExtraConfigFile(modelId), currentConfig)
        }
        onSettingsDoneListener?.let { it(needRecreate) }
    }

    override fun resetSettingsToDefaults() {
        super.resetSettingsToDefaults()
        updateSamplerSettingsVisibility()
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }

    fun setSession(chatSession: LlmSession?) {
        this.chatSession = chatSession
    }

    fun setModelItem(modelItem: ModelItem) {
        this.modelItem = modelItem
    }

    companion object {
        const val TAG = "SettingsBottomSheetFragment"
    }
}
