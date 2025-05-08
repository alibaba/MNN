// Created by ruoyi.sjd on 2025/4/29.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.modelsettings

import android.app.Dialog
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.FrameLayout
import android.widget.SeekBar
import androidx.core.view.isVisible
import androidx.core.widget.addTextChangedListener
import com.alibaba.mnnllm.android.llm.ChatSession
import com.alibaba.mnnllm.android.databinding.FragmentSettingsSheetBinding
import com.alibaba.mnnllm.android.databinding.SettingsRowSliderSwitchBinding
import com.alibaba.mnnllm.android.llm.LlmSession
import com.alibaba.mnnllm.android.utils.UiUtils
import com.google.android.material.bottomsheet.BottomSheetBehavior
import com.google.android.material.bottomsheet.BottomSheetDialogFragment
import java.util.*

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

class SettingsBottomSheetFragment : BottomSheetDialogFragment() {

    private lateinit var loadedConfig: ModelConfig
    private val defaultConfig:ModelConfig = ModelConfig (
        llmModel = "",
        llmWeight = "",
        backendType = "",
        threadNum = 0,
        precision = "",
        memory = "",
        systemPrompt = "You are a helpful assistant.",
        samplerType = "",
        mixedSamplers = mutableListOf(),
        temperature = 0.0f,
        topP = 0.9f,
        topK = 0,
        minP = 0.0f,
        tfsZ = 1.0f,
        typical = 1.0f,
        penalty = 1.02f,
        nGram = 8,
        nGramFactor = 1.02f,
        maxNewTokens = 2048,
        assistantPromptTemplate = ""
    )
    private lateinit var currentConfig:ModelConfig
    private lateinit var chatSession: LlmSession
    private var _binding: FragmentSettingsSheetBinding? = null
    private val binding get() = _binding!!
    private var useMmap: Boolean = true
    private var currentSamplerType: SamplerType = SamplerType.Mixed
    private var penaltySamplerValue: String = "Greedy"

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentSettingsSheetBinding.inflate(inflater, container, false)
        return binding.root
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
        setupModelConfig()
        setupSamplerSettings()
        setupActionButtons()
        setupMaxTokenListener()
        setupSystemPromptListener()
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
            currentConfig.mixedSamplers  = mutableListOf(
                SamplerType.TopK.value,
                SamplerType.TopP.value,
                SamplerType.MinP.value,
                SamplerType.Temperature.value,
            )
        }
        setupSliderSwitchRow(
            rowBinding = SettingsRowSliderSwitchBinding.bind(binding.rowMixedTopK.root),
            label = samplerTypeToString(SamplerType.TopK),
            initialValue = (currentConfig.topK?:defaultConfig.topK!!).toFloat(),
            initialEnabled = currentConfig.mixedSamplers!!.contains(SamplerType.TopK.value),
            valueRange = 1f..100f,
            decimalPlaces = 0,
            onValueChange = { currentConfig.topK = it.toInt() },
            onEnabledChange = { toggleEnable(currentConfig.mixedSamplers!!, SamplerType.TopK, it)}
        )

        setupSliderSwitchRow(
            rowBinding = SettingsRowSliderSwitchBinding.bind(binding.rowMixedTfsZ.root),
            label = samplerTypeToString(SamplerType.Tfs),
            initialValue = (currentConfig.tfsZ?:defaultConfig.tfsZ!!),
            initialEnabled = currentConfig.mixedSamplers!!.contains(SamplerType.Tfs.value),
            valueRange = 0f..1f,
            decimalPlaces = 0,
            onValueChange = { currentConfig.tfsZ = it },
            onEnabledChange = { toggleEnable(currentConfig.mixedSamplers!!, SamplerType.Tfs, it)}
        )

        setupSliderSwitchRow(
            rowBinding = SettingsRowSliderSwitchBinding.bind(binding.rowMixedTypical.root),
            label = samplerTypeToString(SamplerType.Typical),
            initialValue = (currentConfig.typical?:defaultConfig.typical!!).toFloat(),
            initialEnabled = currentConfig.mixedSamplers!!.contains(SamplerType.Typical.value),
            valueRange = 0f..1f,
            decimalPlaces = 0,
            onValueChange = { currentConfig.typical = it },
            onEnabledChange = { toggleEnable(currentConfig.mixedSamplers!!, SamplerType.Typical, it)}
        )

        setupSliderSwitchRow(
            rowBinding = SettingsRowSliderSwitchBinding.bind(binding.rowMixedTopP.root),
            label = samplerTypeToString(SamplerType.TopP),
            initialValue = (currentConfig.topP?:defaultConfig.topP!!).toFloat(),
            initialEnabled = currentConfig.mixedSamplers!!.contains(SamplerType.TopP.value),
            valueRange = 0f..1f,
            decimalPlaces = 2,
            onValueChange = { currentConfig.topP = it },
            onEnabledChange = { toggleEnable(currentConfig.mixedSamplers!!, SamplerType.TopP, it)}
        )

        setupSliderSwitchRow(
            rowBinding = SettingsRowSliderSwitchBinding.bind(binding.rowMixedMinP.root),
            label = samplerTypeToString(SamplerType.MinP),
            initialValue = (currentConfig.minP?:defaultConfig.minP!!).toFloat(),
            initialEnabled = currentConfig.mixedSamplers!!.contains(SamplerType.MinP.value),
            valueRange = 0f..1f,
            decimalPlaces = 2,
            onValueChange = { currentConfig.minP = it },
            onEnabledChange = { toggleEnable(currentConfig.mixedSamplers!!, SamplerType.MinP, it)}
        )

        setupSliderSwitchRow(
            rowBinding = SettingsRowSliderSwitchBinding.bind(binding.rowMixedTemp.root),
            label = samplerTypeToString(SamplerType.Temperature),
            initialValue = (currentConfig.temperature?:defaultConfig.temperature!!).toFloat(),
            initialEnabled = currentConfig.mixedSamplers!!.contains(SamplerType.Temperature.value),
            valueRange = 0f..2f,
            decimalPlaces = 2,
            onValueChange = { currentConfig.temperature = it },
            onEnabledChange = { toggleEnable(currentConfig.mixedSamplers!!, SamplerType.Temperature, it)}
        )
    }

    private fun setupPenaltySettings() {
        setupSliderSwitchRow(
            rowBinding = SettingsRowSliderSwitchBinding.bind(binding.rowPenaltyPenalty.root),
            label = "Penalty",
            initialValue = currentConfig.penalty?:defaultConfig.penalty!!,
            initialEnabled = true,
            valueRange = 0f..5f,
            decimalPlaces = 2,
            onValueChange = { currentConfig.penalty = it },
            switchVisible = false
        )

        setupSliderSwitchRow(
            rowBinding = SettingsRowSliderSwitchBinding.bind(binding.rowPenaltyNgramSize.root),
            label = "N-gram Size",
            initialValue = (currentConfig.nGram?:defaultConfig.nGram!!).toFloat(),
            initialEnabled = true,
            valueRange = 1f..16f,
            decimalPlaces = 0,
            onValueChange = { currentConfig.nGram = it.toInt() },
            switchVisible = false
        )

        setupSliderSwitchRow(
            rowBinding = SettingsRowSliderSwitchBinding.bind(binding.rowPenaltyNgramFactor.root),
            label = "N-gram Factor",
            initialValue = currentConfig.nGramFactor?:defaultConfig.nGramFactor!!,
            initialEnabled = true,
            valueRange = 1f..2f,
            decimalPlaces = 1,
            onValueChange = { currentConfig.nGramFactor = it },
            switchVisible = false
        )
        binding.dropdownPenaltySampler.setDropDownItems(listOf("greedy", "temperature")) { _, value ->
            penaltySamplerValue = value.toString()
        }
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
        switchVisible:Boolean = true
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

    private fun setupActionButtons() {
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

    private fun loadSettings() {
        loadedConfig = chatSession.loadConfig()!!
        currentConfig = loadedConfig.deepCopy()
        updateSamplerSettings()
        //max tokens
        currentConfig.maxNewTokens = currentConfig.maxNewTokens?:defaultConfig.maxNewTokens
        binding.editMaxNewTokens.setText(currentConfig.maxNewTokens.toString())

        //system prompt
        currentConfig.systemPrompt = currentConfig.systemPrompt?:defaultConfig.systemPrompt
        binding.editTextSystemPrompt.setText(currentConfig.systemPrompt)
    }

    private fun saveSettings() {
        var needRecreate = false
        var needSaveConfig = false
        if (currentConfig == loadedConfig) {
            return
        } else if (!currentConfig.samplerEquals(loadedConfig)) {
            needSaveConfig = true
            needRecreate = true
        } else if (currentConfig.maxNewTokens != loadedConfig.maxNewTokens) {
            needSaveConfig = true
            chatSession.updateMaxNewTokens(currentConfig.maxNewTokens!!)
            needRecreate = false
        } else if (currentConfig.systemPrompt != loadedConfig.systemPrompt) {
            needSaveConfig = true
            chatSession.updateSystemPrompt(currentConfig.systemPrompt!!)
            needRecreate = false
        }
        if (needSaveConfig) {
            ModelConfig.saveConfig(chatSession.getModelSettingsFile(), currentConfig)
        }
        if (needRecreate) {
            UiUtils.getActivity(context)?.recreate()
        }
    }

    private fun resetSettingsToDefaults() {
        loadSettings()
        updateSamplerSettingsVisibility()
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }

    fun setSession(chatSession: LlmSession) {
        this.chatSession = chatSession
    }

    companion object {
        const val TAG = "SettingsBottomSheetFragment"
    }
}