package com.alibaba.mnnllm.android.modelsettings // Your package

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ArrayAdapter
import android.widget.SeekBar
import android.widget.Toast
import androidx.core.view.isVisible
import com.alibaba.mnnllm.android.databinding.FragmentSettingsSheetBinding
import com.alibaba.mnnllm.android.databinding.SettingsRowSliderBinding
import com.alibaba.mnnllm.android.databinding.SettingsRowSliderSwitchBinding
import com.google.android.material.bottomsheet.BottomSheetDialogFragment
import java.util.*

enum class SamplerType { Mixed, Penalty, TopP }

class SettingsBottomSheetFragment : BottomSheetDialogFragment() {

    private var _binding: FragmentSettingsSheetBinding? = null
    private val binding get() = _binding!!

    private var useMmap: Boolean = true
    private var currentSamplerType: SamplerType = SamplerType.Mixed
    private var penaltySamplerValue: String = "Greedy"
    private var topKValue: Float = 20f
    private var topKEnabled: Boolean = true
    private var topPValue: Float = 0.84f // Shared state
    private var topPEnabled: Boolean = true
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
        setupSamplerTypeDropdown()
        setupMixedSettings()
        setupPenaltySettings()
        setupTopPSettings()
        updateSamplerSettingsVisibility()
        setupActionButtons()
    }

    private fun setupMixedSettings() {
        setupSliderSwitchRow(
            rowBinding = SettingsRowSliderSwitchBinding.bind(binding.rowMixedTopK.root), // Bind included layout
            label = "Top K",
            initialValue = topKValue,
            initialEnabled = topKEnabled,
            valueRange = 1f..100f,
            decimalPlaces = 0, // Integer value
            onValueChange = { topKValue = it },
            onEnabledChange = { topKEnabled = it }
        )
        setupSliderSwitchRow(
            rowBinding = SettingsRowSliderSwitchBinding.bind(binding.rowMixedTopP.root),
            label = "Top P",
            initialValue = topPValue, // Use shared state
            initialEnabled = topPEnabled,
            valueRange = 0f..1f,
            decimalPlaces = 2,
            onValueChange = { topPValue = it },
            onEnabledChange = { topPEnabled = it }
        )
        setupSliderSwitchRow(
            rowBinding = SettingsRowSliderSwitchBinding.bind(binding.rowMixedTemp.root),
            label = "Temperature",
            initialValue = tempValue,
            initialEnabled = tempEnabled,
            valueRange = 0f..2f, // Example range
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

        // Penalty Sampler Clickable Row
        binding.valuePenaltySampler.text = penaltySamplerValue
        binding.rowPenaltySampler.setOnClickListener {
            // Show Dialog/Bottom Sheet to select Penalty Sampler
            Toast.makeText(requireContext(), "Select Penalty Sampler", Toast.LENGTH_SHORT).show()
            // Update penaltySamplerValue and binding.valuePenaltySampler.text when selected
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

        // --- SeekBar Setup (same as setupSliderRow) ---
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

        // --- Switch Setup ---
        rowBinding.switchSlider.isChecked = initialEnabled
        rowBinding.seekbar.isEnabled = initialEnabled // Initial state for seekbar

        rowBinding.switchSlider.setOnCheckedChangeListener { _, isChecked ->
            rowBinding.seekbar.isEnabled = isChecked
            onEnabledChange(isChecked)
        }
    }


    private fun setupModelConfig() {
        // Use binding directly
        binding.switchUseMmap.isChecked = useMmap // Assuming ID is switch_use_mmap
        binding.switchUseMmap.setOnCheckedChangeListener { _, isChecked ->
            useMmap = isChecked
        }
        binding.buttonClearCache.setOnClickListener { // Assuming ID button_clear_cache
            Toast.makeText(requireContext(), "Clear Cache Clicked", Toast.LENGTH_SHORT).show()
        }
        // Make sure IDs in fragment_settings_sheet.xml match the ones used here
    }

    private fun setupSamplerTypeDropdown() {
        val samplerTypeItems = SamplerType.values().map { samplerTypeToString(it) }
        val adapter = ArrayAdapter(requireContext(), // Use requireContext()
            android.R.layout.simple_dropdown_item_1line, samplerTypeItems)
        val autoCompleteTextView = binding.menuSamplerTypeAutocomplete
        autoCompleteTextView.setAdapter(adapter)
        autoCompleteTextView.setText(samplerTypeToString(currentSamplerType), false) // Set after loadSettings
        autoCompleteTextView.setOnItemClickListener { parent, _, position, _ ->
            val selectedString = parent.getItemAtPosition(position) as String
            val selectedSamplerType = SamplerType.values().find { samplerTypeToString(it) == selectedString }
            if (selectedSamplerType != null && currentSamplerType != selectedSamplerType) {
                currentSamplerType = selectedSamplerType
                updateSamplerSettingsVisibility()
            }
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

    // ... (Implement/Adapt setupMixedSettings, setupPenaltySettings, setupTopPSettings) ...


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
            SamplerType.TopP -> "Top-P"
        }
    }

    private fun updateSamplerSettingsVisibility() {
        // Use binding directly, e.g.:
        binding.containerMixedSettings.isVisible = (currentSamplerType == SamplerType.Mixed)
        binding.containerPenaltySettings.isVisible = (currentSamplerType == SamplerType.Penalty)
        binding.containerTopPSettings.isVisible = (currentSamplerType == SamplerType.TopP)
    }

    // --- State Persistence (Adapt for Fragment context if needed) ---
    private fun loadSettings() {
        // TODO: Load settings - Get context via requireContext() if needed for SharedPreferences
        // Update UI elements using binding.* after loading
        binding.menuSamplerTypeAutocomplete
            .setText(samplerTypeToString(currentSamplerType), false) // Example update
        // ... update all other views based on loaded state ...
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