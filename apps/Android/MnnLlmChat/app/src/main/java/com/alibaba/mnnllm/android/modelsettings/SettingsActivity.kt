// Created by ruoyi.sjd on 2025/4/9.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.modelsettings

import android.os.Bundle
import android.view.Menu
import android.view.MenuItem
import android.view.View
import android.widget.ArrayAdapter
import android.widget.PopupMenu
import android.widget.SeekBar
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.isVisible
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.databinding.ActivitySettingsBinding
import com.alibaba.mnnllm.android.databinding.SettingsRowSliderBinding
import com.alibaba.mnnllm.android.databinding.SettingsRowSliderSwitchBinding
import java.util.Locale

// --- Data Structures ---
enum class SamplerType {
    Mixed, Penalty, TopP
}

class SettingsActivity : AppCompatActivity() {

    private lateinit var binding: ActivitySettingsBinding

    // --- State Variables (Ideally use ViewModel + SharedPreferences/DataStore) ---
    private var useMmap: Boolean = true
    private var currentSamplerType: SamplerType = SamplerType.Mixed
    private var penaltySamplerValue: String = "Greedy"

    // Mixed settings state
    private var topKValue: Float = 20f
    private var topKEnabled: Boolean = true
    // ... (add other mixed state variables: tfsZValue, tfsZEnabled, typicalValue, etc.)
    private var topPValue: Float = 0.84f // Shared state
    private var topPEnabled: Boolean = true
    private var tempValue: Float = 0.82f
    private var tempEnabled: Boolean = true

    // Penalty settings state
    private var penaltyValue: Float = 0.00f
    private var nGramSizeValue: Int = 8
    private var nGramFactorValue: Float = 1.2f


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivitySettingsBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setSupportActionBar(binding.toolbar)
        supportActionBar?.setDisplayHomeAsUpEnabled(true) // Optional: Add back button

        // Load initial state (e.g., from SharedPreferences or ViewModel)
        loadSettings()

        // Setup UI elements and listeners
        setupModelConfig()
        setupSamplerTypeDropdown() // <--- Call the new setup function
//        setupSamplingStrategy()
        setupMixedSettings()
        setupPenaltySettings()
        setupTopPSettings()

        // Set initial visibility based on loaded sampler type
        updateSamplerSettingsVisibility()
    }

    // --- Options Menu (for Done button) ---
    override fun onCreateOptionsMenu(menu: Menu): Boolean {
        menuInflater.inflate(R.menu.settings_menu, menu)
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        return when (item.itemId) {
            android.R.id.home -> { // Handle back button press
                onBackPressedDispatcher.onBackPressed()
                true
            }
            R.id.action_done -> {
                saveSettings() // Save current state
                finish() // Close the activity
                true
            }
            else -> super.onOptionsItemSelected(item)
        }
    }

    // --- Setup Functions ---

    private fun setupModelConfig() {
        binding.switchUseMmap.isChecked = useMmap
        binding.switchUseMmap.setOnCheckedChangeListener { _, isChecked ->
            useMmap = isChecked
            // Optional: Immediately save or handle change
        }

        binding.buttonClearCache.setOnClickListener {
            // Implement cache clearing logic
            Toast.makeText(this, "Clear Cache Clicked", Toast.LENGTH_SHORT).show()
        }
    }


    // --- NEW FUNCTION TO SHOW POPUP MENU ---
//    private fun showSamplerTypePopupMenu(anchorView: View) {
//        val popupMenu = PopupMenu(this, anchorView) // Anchor the popup to the clicked row
//
//        // Add menu items programmatically from the Enum
//        SamplerType.values().forEachIndexed { index, samplerType ->
//            popupMenu.menu.add(Menu.NONE, index, index, samplerTypeToString(samplerType))
//            // We use the enum ordinal as the item ID for simplicity
//        }
//
//        // Set listener for menu item clicks
//        popupMenu.setOnMenuItemClickListener { menuItem ->
//            val selectedSamplerType = SamplerType.values()[menuItem.itemId] // Get enum value using the ID (ordinal)
//
//            // Check if the selection actually changed
//            if (currentSamplerType != selectedSamplerType) {
//                currentSamplerType = selectedSamplerType
//                binding.valueSamplerType.text = samplerTypeToString(currentSamplerType)
//                updateSamplerSettingsVisibility() // Update the dynamic sections below
//                // Optional: Trigger save or other actions if needed immediately
//            }
//            true // Indicate the click was handled
//        }
//
//        // Show the popup menu
//        popupMenu.show()
//    }

//    private fun setupSamplingStrategy() {
//        binding.valueSamplerType.text = samplerTypeToString(currentSamplerType)
//        binding.rowSamplerType.setOnClickListener { view -> // The view that was clicked (the ConstraintLayout row)
//            showSamplerTypePopupMenu(view)
//        }
//    }

    private fun setupSamplerTypeDropdown() {
        // 1. Get the list of display names for the dropdown
        val samplerTypeItems = SamplerType.values().map { samplerTypeToString(it) }

        // 2. Create an ArrayAdapter
        val adapter = ArrayAdapter(
            this,
            android.R.layout.simple_dropdown_item_1line, // Standard layout for dropdown items
            // Alternatively use: R.layout.mtrl_dropdown_menu_popup_item for Material style
            samplerTypeItems
        )

        // 3. Get reference to the AutoCompleteTextView
        val autoCompleteTextView = binding.menuSamplerTypeAutocomplete // No need for cast with view binding

        // 4. Set the adapter
        autoCompleteTextView.setAdapter(adapter)

        // 5. Set the initial value (from loaded settings) - IMPORTANT: use setText with false
        autoCompleteTextView.setText(samplerTypeToString(currentSamplerType), false)

        // 6. Set listener for item selection
        autoCompleteTextView.setOnItemClickListener { parent, view, position, id ->
            val selectedString = parent.getItemAtPosition(position) as String
            // Find the corresponding enum value (can be done more robustly, but this works)
            val selectedSamplerType = SamplerType.values().find { samplerTypeToString(it) == selectedString }

            if (selectedSamplerType != null && currentSamplerType != selectedSamplerType) {
                currentSamplerType = selectedSamplerType
                // No need to update the text view manually, AutoCompleteTextView does it
                updateSamplerSettingsVisibility()
                // Optional: Trigger save or other actions
            }
        }
    }

    private fun setupMixedSettings() {
        // Setup individual rows using the included layout bindings
        // Example for Top K:
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
        // Example for Top P:
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
        // Example for Temperature:
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
        // ... Setup other mixed rows (TFS-Z, Typical, Min P) similarly ...
        // Remember to give IDs to the <include> tags in activity_settings.xml
        // e.g., android:id="@+id/row_mixed_tfs_z"
    }

    private fun setupPenaltySettings() {
        // Penalty Slider
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
            Toast.makeText(this, "Select Penalty Sampler", Toast.LENGTH_SHORT).show()
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

    // --- Helper Functions ---

    private fun samplerTypeToString(type: SamplerType): String {
        return when (type) {
            SamplerType.Mixed -> "Mixed"
            SamplerType.Penalty -> "Penalty"
            SamplerType.TopP -> "Top-P"
        }
    }

    private fun updateSamplerSettingsVisibility() {
        binding.containerMixedSettings.isVisible = (currentSamplerType == SamplerType.Mixed)
        binding.containerPenaltySettings.isVisible = (currentSamplerType == SamplerType.Penalty)
        binding.containerTopPSettings.isVisible = (currentSamplerType == SamplerType.TopP)
    }

    // Helper to configure a Slider row from an included layout
    private fun setupSliderRow(
        rowBinding: SettingsRowSliderBinding, // Use binding for the included layout
        label: String,
        initialValue: Float,
        valueRange: ClosedFloatingPointRange<Float>,
        decimalPlaces: Int,
        onValueChange: (Float) -> Unit
    ) {
        val valueFormat = "%.${decimalPlaces}f"
        rowBinding.labelSlider.text = label
        rowBinding.valueSlider.text = String.format(Locale.US, valueFormat, initialValue)

        // Map float value to SeekBar progress (integer)
        val maxProgress = 1000 // Arbitrary precision
        val range = valueRange.endInclusive - valueRange.start
        rowBinding.seekbar.max = maxProgress
        rowBinding.seekbar.progress = ((initialValue - valueRange.start) / range * maxProgress).toInt()

        rowBinding.seekbar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                if (fromUser) {
                    val newValue = valueRange.start + (progress.toFloat() / maxProgress) * range
                    // Clamp value just in case due to float math
                    val clampedValue = newValue.coerceIn(valueRange)
                    rowBinding.valueSlider.text = String.format(Locale.US, valueFormat, clampedValue)
                    onValueChange(clampedValue)
                }
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })
    }

    // Helper to configure a Slider+Switch row from an included layout
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

    // --- State Persistence (Example Placeholders) ---
    private fun loadSettings() {
        // TODO: Load settings from SharedPreferences, ViewModel, etc.
        // Example:
        // val prefs = getSharedPreferences("AppSettings", MODE_PRIVATE)
        // useMmap = prefs.getBoolean("useMmap", true)
        // currentSamplerType = SamplerType.valueOf(prefs.getString("samplerType", SamplerType.Mixed.name) ?: SamplerType.Mixed.name)
        // topKValue = prefs.getFloat("topK", 20f)
        // ... load other values ...

        // Update UI based on loaded values after loading
        binding.switchUseMmap.isChecked = useMmap
//        binding.valueSamplerType.text = samplerTypeToString(currentSamplerType)
        // ... update seekbars, other switches, text values ...
    }

    private fun saveSettings() {
        // TODO: Save settings to SharedPreferences, ViewModel, etc.
        // Example:
        // val prefs = getSharedPreferences("AppSettings", MODE_PRIVATE).edit()
        // prefs.putBoolean("useMmap", useMmap)
        // prefs.putString("samplerType", currentSamplerType.name)
        // prefs.putFloat("topK", topKValue)
        // ... save other values ...
        // prefs.apply()
        Toast.makeText(this, "Settings Saved (Placeholder)", Toast.LENGTH_SHORT).show()
    }
}