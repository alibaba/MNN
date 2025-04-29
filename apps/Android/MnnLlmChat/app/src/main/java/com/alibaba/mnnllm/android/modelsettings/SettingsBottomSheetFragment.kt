// Created by ruoyi.sjd on 2025/4/29.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.modelsettings

import android.os.Bundle
import android.view.LayoutInflater
import android.view.Menu
import android.view.View
import android.view.ViewGroup
import android.widget.SeekBar
import android.widget.Toast
import androidx.appcompat.widget.PopupMenu
import androidx.core.view.isVisible
import com.alibaba.mnnllm.android.databinding.FragmentSettingsSheetBinding
import com.alibaba.mnnllm.android.databinding.SettingsRowSliderBinding
import com.alibaba.mnnllm.android.databinding.SettingsRowSliderSwitchBinding
import com.google.android.material.bottomsheet.BottomSheetDialogFragment
import java.util.*

enum class SamplerType { Greedy, Temperature, TopK, TopP, MinP, Tfs, Typical, Penalty, Mixed}

class SettingsBottomSheetFragment : BottomSheetDialogFragment() {

    private var _binding: FragmentSettingsSheetBinding? = null
    private val binding get() = _binding!!
    private var useMmap: Boolean = true
    private var currentSamplerType: SamplerType = SamplerType.Mixed
    private var penaltySamplerValue: String = "Greedy"

    private var topKValue: Float = 40f
    private var topKEnabled: Boolean = false

    private var topPValue: Float = 0.9f
    private var topPEnabled: Boolean = false

    private var minPValue: Float = 0.9f
    private var minPEnabled: Boolean = false

    private var tfsZValue: Float = 1.0f
    private var tfsZEnabled: Boolean = false

    private var typicalValue: Float = 1.0f
    private var typicalEnabled: Boolean = false

    private var tempValue: Float = 0.82f
    private var tempEnabled: Boolean = true

    private var penaltyValue: Float = 0.00f
    private var nGramSizeValue: Int = 8
    private var nGramFactorValue: Float = 1.2f

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentSettingsSheetBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        loadSettings()
        setupModelConfig()
        setupMixedSettings()
        setupPenaltySettings()
        setupTopPSettings()
        updateSamplerSettingsVisibility()
        setupActionButtons()
    }

    private fun setupMixedSettings() {
        setupSliderSwitchRow(
            rowBinding = SettingsRowSliderSwitchBinding.bind(binding.rowMixedTopK.root),
            label = samplerTypeToString(SamplerType.TopK), // Use samplerTypeToString
            initialValue = topKValue,
            initialEnabled = topKEnabled,
            valueRange = 1f..100f,
            decimalPlaces = 0, // Integer value
            onValueChange = { topKValue = it },
            onEnabledChange = { topKEnabled = it }
        )

        setupSliderSwitchRow(
            rowBinding = SettingsRowSliderSwitchBinding.bind(binding.rowMixedTfsZ.root),
            label = samplerTypeToString(SamplerType.Tfs), // Use samplerTypeToString
            initialValue = tfsZValue,
            initialEnabled = tfsZEnabled,
            valueRange = 0f..1f,
            decimalPlaces = 0,
            onValueChange = { tfsZValue = it },
            onEnabledChange = { tfsZEnabled = it }
        )

        setupSliderSwitchRow(
            rowBinding = SettingsRowSliderSwitchBinding.bind(binding.rowMixedTypical.root),
            label = samplerTypeToString(SamplerType.Typical), // Use samplerTypeToString
            initialValue = typicalValue,
            initialEnabled = typicalEnabled,
            valueRange = 0f..1f,
            decimalPlaces = 0,
            onValueChange = { typicalValue = it },
            onEnabledChange = { typicalEnabled = it }
        )

        setupSliderSwitchRow(
            rowBinding = SettingsRowSliderSwitchBinding.bind(binding.rowMixedTopP.root),
            label = samplerTypeToString(SamplerType.TopP), // Use samplerTypeToString
            initialValue = topPValue, // Use shared state
            initialEnabled = topPEnabled,
            valueRange = 0f..1f,
            decimalPlaces = 2,
            onValueChange = { topPValue = it },
            onEnabledChange = { topPEnabled = it }
        )

        setupSliderSwitchRow(
            rowBinding = SettingsRowSliderSwitchBinding.bind(binding.rowMixedMinP.root),
            label = samplerTypeToString(SamplerType.MinP), // Use samplerTypeToString
            initialValue = minPValue, // Use shared state
            initialEnabled = minPEnabled,
            valueRange = 0f..1f,
            decimalPlaces = 2,
            onValueChange = { minPValue = it },
            onEnabledChange = { minPEnabled = it }
        )

        setupSliderSwitchRow(
            rowBinding = SettingsRowSliderSwitchBinding.bind(binding.rowMixedTemp.root),
            label = samplerTypeToString(SamplerType.Temperature), // Use samplerTypeToString
            initialValue = tempValue,
            initialEnabled = tempEnabled,
            valueRange = 0f..2f,
            decimalPlaces = 2,
            onValueChange = { tempValue = it },
            onEnabledChange = { tempEnabled = it }
        )
    }

    private fun setupPenaltySettings() {
        setupSliderRow(
            rowBinding = SettingsRowSliderBinding.bind(binding.rowPenaltyPenalty.root),
            label = "Penalty",
            initialValue = penaltyValue,
            valueRange = 0f..5f, // Example range
            decimalPlaces = 2,
            onValueChange = { penaltyValue = it }
        )

        // N-gram Size Slider (Integer)
        setupSliderRow(
            rowBinding = SettingsRowSliderBinding.bind(binding.rowPenaltyNgramSize.root),
            label = "N-gram Size",
            initialValue = nGramSizeValue.toFloat(), // SeekBar needs float
            valueRange = 1f..16f, // Example range
            decimalPlaces = 0,
            onValueChange = { nGramSizeValue = it.toInt() } // Convert back to Int
        )

        // N-gram Factor Slider
        setupSliderRow(
            rowBinding = SettingsRowSliderBinding.bind(binding.rowPenaltyNgramFactor.root),
            label = "N-gram Factor",
            initialValue = nGramFactorValue,
            valueRange = 0f..5f, // Example range
            decimalPlaces = 1, // Example precision
            onValueChange = { nGramFactorValue = it }
        )

        binding.valuePenaltySampler.text = penaltySamplerValue
        binding.rowPenaltySampler.setOnClickListener {
            Toast.makeText(requireContext(), "Select Penalty Sampler", Toast.LENGTH_SHORT).show()
        }
    }

    private fun setupTopPSettings() {
        // Top P Slider (reuse shared state)
        setupSliderRow(
            rowBinding = SettingsRowSliderBinding.bind(binding.rowTopPTopP.root),
            label = "Top P",
            initialValue = topPValue,
            valueRange = 0f..1f,
            decimalPlaces = 2,
            onValueChange = { topPValue = it }
        )
    }

    private fun setupSliderSwitchRow(
        rowBinding: SettingsRowSliderSwitchBinding, // Use binding for the included layout
        label: String,
        initialValue: Float,
        initialEnabled: Boolean,
        valueRange: ClosedFloatingPointRange<Float>,
        decimalPlaces: Int,
        onValueChange: (Float) -> Unit,
        onEnabledChange: (Boolean) -> Unit
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
        rowBinding.seekbar.isEnabled = initialEnabled // Initial state for seekbar

        rowBinding.switchSlider.setOnCheckedChangeListener { _, isChecked ->
            rowBinding.seekbar.isEnabled = isChecked
            onEnabledChange(isChecked)
        }
    }

    private fun setupModelConfig() {
        binding.switchUseMmap.isChecked = useMmap // Assuming ID is switch_use_mmap
        binding.switchUseMmap.setOnCheckedChangeListener { _, isChecked ->
            useMmap = isChecked
        }
    }

    // Adapt setupSliderRow, setupSliderSwitchRow etc. to use 'binding' and 'requireContext()'
    // Example adaptation:
    private fun setupSliderRow(
        rowBinding: SettingsRowSliderBinding, // Or access directly via binding.rowId...
        label: String,
        initialValue: Float,
        valueRange: ClosedFloatingPointRange<Float>,
        decimalPlaces: Int,
        onValueChange: (Float) -> Unit
    ) {
        // If using included layouts, you might bind them like this:
        // val sliderBinding = SettingsRowSliderBinding.bind(rowBinding.root) // Pass the root view of the included layout
        // Or access views directly if not using includes: e.g., binding.seekbarPenalty
        // ... implementation using the correct view references ...
    }


    private fun setupActionButtons() {
        binding.buttonCancel.setOnClickListener {
            dismiss() // Dismiss the bottom sheet
        }
//        binding.buttonCloseSheet.setOnClickListener { // Handle optional close button
//            dismiss()
//        }
        binding.buttonSave.setOnClickListener {
            saveSettings() // Call your save logic
            dismiss() // Dismiss after saving
        }
        binding.buttonReset.setOnClickListener {
            // Implement reset logic: reset state variables, update UI
            resetSettingsToDefaults() // Example function call
            Toast.makeText(requireContext(), "Settings Reset (Placeholder)", Toast.LENGTH_SHORT).show()
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
        binding.rlSamplerType.setOnClickListener{v->
            showSamplerTypePopupMenu(binding.tvSamplerTypeValue)
        }
    }

    private fun showSamplerTypePopupMenu(anchorView: View) {
        val popupMenu = PopupMenu(anchorView.context, anchorView)
        SamplerType.entries.forEachIndexed { index, samplerType ->
            popupMenu.menu.add(Menu.NONE, index, index, samplerTypeToString(samplerType))
        }
        popupMenu.setOnMenuItemClickListener { menuItem ->
            val selectedSamplerType = SamplerType.entries[menuItem.itemId]
            if (currentSamplerType != selectedSamplerType) {
                currentSamplerType = selectedSamplerType
                binding.tvSamplerTypeValue.text = samplerTypeToString(currentSamplerType)
                updateSamplerSettingsVisibility()
            }
            true
        }
        popupMenu.show()
    }

    private fun loadSettings() {

    }

    private fun saveSettings() {
        // TODO: Save settings - Get context via requireContext() if needed
        Toast.makeText(requireContext(), "Settings Saved (Placeholder)", Toast.LENGTH_SHORT).show()
    }

    private fun resetSettingsToDefaults() {
        // TODO: Implement logic to reset all state variables to default values
        // Example:
        currentSamplerType = SamplerType.Mixed
        topKValue = 20f
        // ... reset all others ...

        // Reload/Update UI after resetting state
        loadSettings() // Easiest way is often to just reload defaults and update UI
        updateSamplerSettingsVisibility()
    }


    // --- Lifecycle ---
    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null // Important to release binding reference
    }

    // --- Companion Object for Tag ---
    companion object {
        const val TAG = "SettingsBottomSheetFragment" // Tag for FragmentManager
    }
}