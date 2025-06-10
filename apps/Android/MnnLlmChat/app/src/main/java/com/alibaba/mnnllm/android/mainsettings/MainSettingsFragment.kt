// Created by ruoyi.sjd on 2025/2/28.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.mainsettings
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.preference.ListPreference
import androidx.preference.Preference
import androidx.preference.PreferenceFragmentCompat
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.update.UpdateChecker
import com.alibaba.mnnllm.android.utils.AppUtils
import com.alibaba.mnnllm.android.utils.PreferenceUtils
import com.alibaba.mnnllm.api.openai.service.ApiServerConfig
import com.alibaba.mnnllm.api.openai.manager.ApiServiceManager

class MainSettingsFragment : PreferenceFragmentCompat() {

    companion object {
        const val TAG = "MainSettingsFragment"
    }

    private var updateChecker: UpdateChecker? = null

    override fun onResume() {
        super.onResume()
        updateChecker?.checkForUpdates(requireContext(), false)
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
                updateChecker = UpdateChecker(requireContext())
                updateChecker?.checkForUpdates(requireContext(), true)
                true
            }
        }

        // 重置 API配置
        val resetApiConfigPref = findPreference<Preference>("reset_api_config")
        resetApiConfigPref?.setOnPreferenceClickListener {
            // 显示 配置确认对话框
            androidx.appcompat.app.AlertDialog.Builder(requireContext())
                .setTitle(R.string.reset_api_config)
                .setMessage(R.string.reset_api_config_confirm_message)
                .setPositiveButton(android.R.string.ok) { _, _ ->
                    // 重置 API配置
                    ApiServerConfig.resetToDefault(requireContext())
                    
                    // 如果API服务正在运行，则使用新配置重启服务
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


        val downloadProviderPref = findPreference<ListPreference>("download_provider")
        downloadProviderPref?.apply {
            fun updateSummary(vale:String) {
                summary = when (vale) {
                    "HuggingFace" -> vale
                    "ModelScope" -> getString(R.string.modelscope)
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
                    "1" -> getString(R.string.diffusion_mode_memory_engough)
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
}