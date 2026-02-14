// Created by ruoyi.sjd on 2025/2/28.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.mainsettings
import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.widget.Toast
import androidx.preference.ListPreference
import androidx.preference.Preference
import androidx.preference.PreferenceFragmentCompat
import androidx.preference.SwitchPreferenceCompat
import com.alibaba.mls.api.source.ModelSources
import com.alibaba.mnnllm.android.MnnLlmApplication
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.MNN
import com.alibaba.mnnllm.android.debug.DebugActivity
import com.alibaba.mnnllm.android.privacy.PrivacyPolicyManager
import com.alibaba.mnnllm.android.update.UpdateChecker
import com.alibaba.mnnllm.android.utils.AppUtils
import com.alibaba.mnnllm.android.utils.PreferenceUtils
import com.alibaba.mnnllm.api.openai.service.ApiServerConfig
import com.alibaba.mnnllm.api.openai.manager.ApiServiceManager
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.alibaba.mnnllm.android.modelmarket.ModelRepository

class MainSettingsFragment : PreferenceFragmentCompat() {

    companion object {
        const val TAG = "MainSettingsFragment"
        private const val DEBUG_CLICK_COUNT = 5
        private const val DEBUG_CLICK_TIMEOUT = 3000L // 3 seconds
    }

    private var updateChecker: UpdateChecker? = null
    private var debugClickCount = 0
    private var debugClickHandler = Handler(Looper.getMainLooper())
    private var debugClickRunnable: Runnable? = null
    private var updateCheckRunnable: Runnable? = null
    private var debugModePref: Preference? = null

    override fun onResume() {
        super.onResume()
        updateChecker?.checkForUpdates(requireContext(), false)
    }


    override fun onStart() {
        super.onStart()

        // Setup debug mode preference
        debugModePref = findPreference<Preference>("debug_mode")
        debugModePref?.setOnPreferenceClickListener {
            val intent = Intent(requireContext(), DebugActivity::class.java)
            startActivity(intent)
            true
        }

        // Ensure debug mode preference is hidden by default unless previously activated
        val sharedPreferences = preferenceManager.sharedPreferences
        val isDebugModeActivated = sharedPreferences?.getBoolean("debug_mode_activated", false) ?: false
        debugModePref?.isVisible = isDebugModeActivated
    }

    override fun onCreatePreferences(savedInstanceState: Bundle?, rootKey: String?) {
        setPreferencesFromResource(R.xml.main_settings_prefs, rootKey)

        val checkUpdatePref = findPreference<Preference>("check_update")
        checkUpdatePref?.apply {
            summary = getString(
                R.string.current_version,
                AppUtils.getAppVersionName(requireContext())
            )
            setOnPreferenceClickListener {
                handleDebugClick()
                updateCheckRunnable?.let { debugClickHandler.removeCallbacks(it) }
                updateCheckRunnable = Runnable {
                    updateChecker = UpdateChecker(requireContext())
                    updateChecker?.checkForUpdates(requireContext(), true)
                }
                debugClickHandler.postDelayed(updateCheckRunnable!!, 1000L)
                true
            }
        }

        // Setup MNN Version preference
        val mnnVersionPref = findPreference<Preference>("mnn_version")
        mnnVersionPref?.apply {
            try {
                val version = MNN.getVersion()
                summary = getString(R.string.mnn_version_summary, version)
            } catch (e: Exception) {
                summary = "N/A"
            }
        }


        //Reset API configuration
        val resetApiConfigPref = findPreference<Preference>("reset_api_config")
        resetApiConfigPref?.setOnPreferenceClickListener {
            MaterialAlertDialogBuilder(requireContext())
                .setTitle(R.string.reset_api_config)
                .setMessage(R.string.reset_api_config_confirm_message)
                .setPositiveButton(android.R.string.ok) { _, _ ->
                    ApiServerConfig.resetToDefault(requireContext())
                    
                    if (MainSettings.isApiServiceEnabled(requireContext()) && ApiServiceManager.isApiServiceRunning()) {
                        ApiServiceManager.stopApiService(requireContext())
                        ApiServiceManager.startApiService(requireContext())
                        Toast.makeText(requireContext(), getString(R.string.api_config_reset_service_restarted), Toast.LENGTH_LONG).show()
                    } else {
                        Toast.makeText(requireContext(), getString(R.string.api_config_reset_to_default), Toast.LENGTH_LONG).show()
                    }
                }
                .setNegativeButton(android.R.string.cancel, null)
                .show()
            true
        }

        val crashDiagnosticsPref = findPreference<SwitchPreferenceCompat>("crash_diagnostics_enabled")
        crashDiagnosticsPref?.apply {
            val privacyManager = PrivacyPolicyManager.getInstance(requireContext())
            isChecked = privacyManager.isCrashReportingConsented()
            setOnPreferenceChangeListener { preference, newValue ->
                val enabled = newValue as Boolean
                if (enabled) {
                    privacyManager.setUserConsent(consented = true)
                    (requireActivity().application as? MnnLlmApplication)?.applyCrashReportingConsent()
                    Toast.makeText(requireContext(), getString(R.string.privacy_policy_consent_enabled), Toast.LENGTH_LONG).show()
                    true
                } else {
                    MaterialAlertDialogBuilder(requireContext())
                        .setTitle(R.string.crash_diagnostics_disable_title)
                        .setMessage(R.string.crash_diagnostics_disable_confirm_message)
                        .setPositiveButton(R.string.crash_diagnostics_disable_confirm_action) { _, _ ->
                            privacyManager.setUserConsent(consented = false)
                            (requireActivity().application as? MnnLlmApplication)?.applyCrashReportingConsent()
                            (preference as SwitchPreferenceCompat).isChecked = false
                            Toast.makeText(requireContext(), getString(R.string.privacy_policy_consent_disabled), Toast.LENGTH_LONG).show()
                        }
                        .setNegativeButton(android.R.string.cancel, null)
                        .show()
                    false
                }
            }
        }

        
        val voiceModelManagementPref = findPreference<Preference>("voice_model_management")
        voiceModelManagementPref?.setOnPreferenceClickListener {
            val voiceModelMarketBottomSheet = com.alibaba.mnnllm.android.chat.voice.VoiceModelMarketBottomSheet.newInstance()
            voiceModelMarketBottomSheet.show(childFragmentManager, "voice_model_market")
            true
        }


        val downloadProviderPref = findPreference<ListPreference>("download_provider")
        downloadProviderPref?.apply {
            fun updateSummary(vale:String) {
                summary = when (vale) {
                    ModelSources.sourceHuffingFace -> vale
                    ModelSources.sourceModelScope -> getString(R.string.modelscope)
                    else -> getString(R.string.modelers)
                }
            }
            preferenceManager.sharedPreferences?.let { sharedPreferences ->
                val defaultProvider = MainSettings.getDownloadProviderString(requireContext())
                if (!sharedPreferences.contains("download_provider")) {
                    sharedPreferences.edit().putString("download_provider", defaultProvider).apply()
                    downloadProviderPref.value = defaultProvider
                }
                updateSummary(value?:defaultProvider)
                onPreferenceChangeListener = Preference.OnPreferenceChangeListener { _, newValue ->
                    updateSummary(newValue.toString())
                    // 同步更新 ModelSources 的下载源类型
                    val sourceType = when (newValue.toString()) {
                        ModelSources.sourceHuffingFace -> ModelSources.ModelSourceType.HUGGING_FACE
                        ModelSources.sourceModelScope -> ModelSources.ModelSourceType.MODEL_SCOPE
                        else -> ModelSources.ModelSourceType.MODELERS
                    }
                    ModelSources.setSourceType(sourceType)
                    // 清除 ModelRepository 缓存以触发重新处理 modelId
                    ModelRepository.clear()
                    Toast.makeText(context, R.string.settings_complete, Toast.LENGTH_LONG).show()
                    true
                }
            }
        }

        // Setup diffusion memory mode preference
        val diffusionMemoryModePref = findPreference<ListPreference>("diffusion_memory_mode")
        diffusionMemoryModePref?.apply {
            fun updateMemorySummary(vale:String) {
                Log.d(TAG, "diffusionMemoryModePref updateSummary vale: $vale")
                diffusionMemoryModePref.summary = when (vale) {
                    "0" -> getString(R.string.diffusion_mode_memory_saving)
                    "1" -> getString(R.string.diffusion_mode_memory_enough)
                    else -> getString(R.string.diffusion_mode_memory_balance)
                }
            }
            val defaultMemoryMode = MainSettings.getDiffusionMemoryMode(requireContext())
            updateMemorySummary(defaultMemoryMode)
            onPreferenceChangeListener = Preference.OnPreferenceChangeListener { _, newValue ->
                val memoryMode = (newValue as String)
                updateMemorySummary(memoryMode)
                MainSettings.setDiffusionMemoryMode(requireContext(), memoryMode)
                true
            }
        }
    }

    private fun handleDebugClick() {
        debugClickCount++
        
        debugClickRunnable?.let { debugClickHandler.removeCallbacks(it) }
        
        if (debugClickCount >= DEBUG_CLICK_COUNT) {
            updateCheckRunnable?.let { debugClickHandler.removeCallbacks(it) }
            // Show debug mode preference instead of directly opening DebugActivity
            debugModePref?.isVisible = true
            // Save debug mode activation state to SharedPreferences
            preferenceManager.sharedPreferences?.edit()?.putBoolean("debug_mode_activated", true)?.apply()
            debugClickCount = 0
            Log.d(TAG, "Debug mode preference activated")
        } else {
            debugClickRunnable = Runnable {
                debugClickCount = 0
                Log.d(TAG, "Debug click count reset due to timeout")
            }
            debugClickHandler.postDelayed(debugClickRunnable!!, DEBUG_CLICK_TIMEOUT)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        debugClickRunnable?.let { debugClickHandler.removeCallbacks(it) }
        updateCheckRunnable?.let { debugClickHandler.removeCallbacks(it) }
    }
}
