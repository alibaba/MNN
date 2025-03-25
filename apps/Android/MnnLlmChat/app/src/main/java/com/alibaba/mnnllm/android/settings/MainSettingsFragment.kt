// Created by ruoyi.sjd on 2025/2/28.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.settings
import android.os.Bundle
import android.widget.Toast
import androidx.preference.ListPreference
import androidx.preference.Preference
import androidx.preference.PreferenceFragmentCompat
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.update.UpdateChecker
import com.alibaba.mnnllm.android.utils.AppUtils
import com.alibaba.mnnllm.android.utils.PreferenceUtils

class MainSettingsFragment : PreferenceFragmentCompat() {
    override fun onCreatePreferences(savedInstanceState: Bundle?, rootKey: String?) {
        setPreferencesFromResource(R.xml.main_settings_prefs, rootKey)

        val checkUpdatePref = findPreference<Preference>("check_update")
        checkUpdatePref?.apply {
            summary = getString(
                R.string.current_version,
                AppUtils.getAppVersionName(requireContext())
            )
            setOnPreferenceClickListener {
                UpdateChecker(requireContext()).checkForUpdates(requireContext(), true)
                true
            }
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
                val defaultProvider =
                    if (!PreferenceUtils.isUseModelsScopeDownload(requireContext())) "HuggingFace" else "ModelScope"
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
    }
}