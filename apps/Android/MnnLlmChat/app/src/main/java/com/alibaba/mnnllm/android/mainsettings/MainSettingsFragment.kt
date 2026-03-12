// Created by ruoyi.sjd on 2025/2/28.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.mainsettings

import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.fragment.app.Fragment
import androidx.preference.PreferenceManager
import com.alibaba.mls.api.source.ModelSources
import com.alibaba.mnnllm.android.MNN
import com.alibaba.mnnllm.android.MnnLlmApplication
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.databinding.FragmentMainSettingsBinding
import com.alibaba.mnnllm.android.debug.DebugActivity
import com.alibaba.mnnllm.android.modelmarket.ModelRepository
import com.alibaba.mnnllm.android.privacy.PrivacyPolicyManager
import com.alibaba.mnnllm.android.update.UpdateChecker
import com.alibaba.mnnllm.android.utils.AppUtils
import com.alibaba.mnnllm.api.openai.manager.ApiServiceManager
import com.alibaba.mnnllm.api.openai.service.ApiServerConfig
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.coroutines.delay

class MainSettingsFragment : Fragment() {

    companion object {
        const val TAG = "MainSettingsFragment"
        private const val DEBUG_CLICK_COUNT = 5
        private const val DEBUG_CLICK_TIMEOUT = 3000L // 3 seconds
    }

    private var _binding: FragmentMainSettingsBinding? = null
    private val binding get() = _binding!!

    private var updateChecker: UpdateChecker? = null
    private var debugClickCount = 0
    private val debugClickHandler = Handler(Looper.getMainLooper())
    private var debugClickRunnable: Runnable? = null
    private var updateCheckRunnable: Runnable? = null

    private var suppressCrashDiagnosticsCallback = false
    private var crashDiagnosticsToggleInProgress = false
    private var crashDiagnosticsDialogShowing = false

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentMainSettingsBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        setupSettings()
    }

    override fun onResume() {
        super.onResume()
        updateChecker?.checkForUpdates(requireContext(), false)
    }

    private fun setupSettings() {
        val sharedPreferences = PreferenceManager.getDefaultSharedPreferences(requireContext())

        binding.itemStopDownload.isChecked = sharedPreferences.getBoolean("stop_download_on_chat", true)
        binding.itemStopDownload.setOnCheckedChangeListener { isChecked ->
            sharedPreferences.edit().putBoolean("stop_download_on_chat", isChecked).apply()
        }

        setupDownloadProvider(sharedPreferences)
        setupVoiceModelManagement()
        setupStorageManagement()

        binding.itemEnableApi.isChecked = MainSettings.isApiServiceEnabled(requireContext())
        binding.itemEnableApi.setOnCheckedChangeListener { isChecked ->
            sharedPreferences.edit().putBoolean("enable_api_service", isChecked).apply()
        }

        setupCrashDiagnostics()
        setupResetApiConfig()
        setupUpdateAndVersion()
        setupDebugMode(sharedPreferences)
    }

    private fun setupDownloadProvider(
        sharedPreferences: android.content.SharedPreferences
    ) {
        val providers = listOf(
            ModelSources.sourceHuffingFace,
            ModelSources.sourceModelScope,
            "Modelers"
        )

        fun providerLabel(provider: String): String {
            return when (provider) {
                ModelSources.sourceHuffingFace -> provider
                ModelSources.sourceModelScope -> getString(R.string.modelscope)
                else -> getString(R.string.modelers)
            }
        }

        val defaultProvider = MainSettings.getDownloadProviderString(requireContext())
        if (!sharedPreferences.contains("download_provider")) {
            sharedPreferences.edit().putString("download_provider", defaultProvider).apply()
        }
        val currentProvider = MainSettings.getDownloadProviderString(requireContext())

        binding.dropdownDownloadProvider.setDropDownItems(
            providers,
            itemToString = { providerLabel(it as String) },
            onDropdownItemSelected = { _, selected ->
                val provider = selected as String
                MainSettings.setDownloadProvider(requireContext(), provider)
                val sourceType = when (provider) {
                    ModelSources.sourceHuffingFace -> ModelSources.ModelSourceType.HUGGING_FACE
                    ModelSources.sourceModelScope -> ModelSources.ModelSourceType.MODEL_SCOPE
                    else -> ModelSources.ModelSourceType.MODELERS
                }
                ModelSources.setSourceType(sourceType)
                ModelRepository.clear()
                Toast.makeText(context, R.string.settings_complete, Toast.LENGTH_LONG).show()
            }
        )
        binding.dropdownDownloadProvider.setCurrentItem(currentProvider)
    }

    private fun setupVoiceModelManagement() {
        binding.btnVoiceModelManagement.setOnClickListener {
            val sheet = com.alibaba.mnnllm.android.chat.voice.VoiceModelMarketBottomSheet.newInstance()
            sheet.show(childFragmentManager, "voice_model_market")
        }
    }

    private fun setupStorageManagement() {
        binding.btnStorageManagement.setOnClickListener {
            startActivity(Intent(requireContext(), StorageManagementActivity::class.java))
        }
    }

    private fun setupResetApiConfig() {
        binding.btnResetApi.setOnClickListener {
            MaterialAlertDialogBuilder(requireContext())
                .setTitle(R.string.reset_api_config)
                .setMessage(R.string.reset_api_config_confirm_message)
                .setPositiveButton(android.R.string.ok) { _, _ ->
                    ApiServerConfig.resetToDefault(requireContext())
                    if (MainSettings.isApiServiceEnabled(requireContext()) && ApiServiceManager.isApiServiceRunning()) {
                        // Run stop/start off main thread to avoid ANR
                        lifecycleScope.launch {
                            withContext(Dispatchers.IO) {
                                ApiServiceManager.stopApiService(requireContext())
                                delay(500)
                                ApiServiceManager.startApiService(requireContext())
                            }
                            Toast.makeText(
                                requireContext(),
                                getString(R.string.api_config_reset_service_restarted),
                                Toast.LENGTH_LONG
                            ).show()
                        }
                    } else {
                        Toast.makeText(
                            requireContext(),
                            getString(R.string.api_config_reset_to_default),
                            Toast.LENGTH_LONG
                        ).show()
                    }
                }
                .setNegativeButton(android.R.string.cancel, null)
                .show()
        }
    }

    private fun setupCrashDiagnostics() {
        val privacyManager = PrivacyPolicyManager.getInstance(requireContext())
        setCrashDiagnosticsChecked(privacyManager.isCrashReportingConsented())

        binding.itemCrashDiagnostics.setOnCheckedChangeListener { isChecked ->
            if (suppressCrashDiagnosticsCallback || crashDiagnosticsToggleInProgress) {
                return@setOnCheckedChangeListener
            }
            crashDiagnosticsToggleInProgress = true
            binding.root.post { crashDiagnosticsToggleInProgress = false }

            if (isChecked) {
                privacyManager.setUserConsent(consented = true)
                (requireActivity().application as? MnnLlmApplication)?.applyCrashReportingConsent()
                Toast.makeText(
                    requireContext(),
                    getString(R.string.privacy_policy_consent_enabled),
                    Toast.LENGTH_LONG
                ).show()
                return@setOnCheckedChangeListener
            }

            if (crashDiagnosticsDialogShowing) {
                return@setOnCheckedChangeListener
            }
            crashDiagnosticsDialogShowing = true

            MaterialAlertDialogBuilder(requireContext())
                .setTitle(R.string.crash_diagnostics_disable_title)
                .setMessage(R.string.crash_diagnostics_disable_confirm_message)
                .setPositiveButton(R.string.crash_diagnostics_disable_confirm_action) { _, _ ->
                    privacyManager.setUserConsent(consented = false)
                    (requireActivity().application as? MnnLlmApplication)?.applyCrashReportingConsent()
                    setCrashDiagnosticsChecked(false)
                    Toast.makeText(
                        requireContext(),
                        getString(R.string.privacy_policy_consent_disabled),
                        Toast.LENGTH_LONG
                    ).show()
                }
                .setNegativeButton(android.R.string.cancel) { _, _ ->
                    setCrashDiagnosticsChecked(true)
                }
                .setOnDismissListener {
                    crashDiagnosticsDialogShowing = false
                }
                .show()
        }
    }

    private fun setCrashDiagnosticsChecked(checked: Boolean) {
        suppressCrashDiagnosticsCallback = true
        binding.itemCrashDiagnostics.isChecked = checked
        suppressCrashDiagnosticsCallback = false
    }

    private fun setupUpdateAndVersion() {
        if (com.alibaba.mnnllm.android.BuildConfig.IS_GOOGLE_PLAY_BUILD) {
            binding.btnCheckUpdate.isClickable = false
            binding.btnCheckUpdate.text = getString(R.string.current_version, AppUtils.getAppVersionName(requireContext()))
        } else {
            binding.btnCheckUpdate.text = getString(
                R.string.current_version_check_update,
                AppUtils.getAppVersionName(requireContext())
            )
            binding.btnCheckUpdate.setOnClickListener {
                handleDebugClick()
                updateCheckRunnable?.let { debugClickHandler.removeCallbacks(it) }
                updateCheckRunnable = Runnable {
                    updateChecker = UpdateChecker(requireContext())
                    updateChecker?.checkForUpdates(requireContext(), true)
                }
                debugClickHandler.postDelayed(updateCheckRunnable!!, 1000L)
            }
        }

        try {
            val version = MNN.getVersion()
            binding.tvMnnVersion.text = version
        } catch (_: Exception) {
            binding.tvMnnVersion.text = "N/A"
        }
    }

    private fun setupDebugMode(
        sharedPreferences: android.content.SharedPreferences
    ) {
        val isActivated = sharedPreferences.getBoolean("debug_mode_activated", false)
        updateDebugModeVisibility(isActivated)

        binding.btnDebugMode.setOnClickListener {
            startActivity(Intent(requireContext(), DebugActivity::class.java))
        }
    }

    private fun handleDebugClick() {
        debugClickCount++

        debugClickRunnable?.let { debugClickHandler.removeCallbacks(it) }

        if (debugClickCount >= DEBUG_CLICK_COUNT) {
            updateCheckRunnable?.let { debugClickHandler.removeCallbacks(it) }
            updateDebugModeVisibility(true)
            PreferenceManager.getDefaultSharedPreferences(requireContext()).edit()
                .putBoolean("debug_mode_activated", true)
                .apply()
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

    private fun updateDebugModeVisibility(visible: Boolean) {
        binding.dividerDebug.visibility = if (visible) View.VISIBLE else View.GONE
        binding.btnDebugMode.visibility = if (visible) View.VISIBLE else View.GONE
    }

    override fun onDestroyView() {
        super.onDestroyView()
        debugClickRunnable?.let { debugClickHandler.removeCallbacks(it) }
        updateCheckRunnable?.let { debugClickHandler.removeCallbacks(it) }
        _binding = null
    }
}
