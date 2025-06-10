// Created by ruoyi.sjd on 2025/3/26.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.taobao.meta.avatar.settings
import android.os.Bundle
import androidx.preference.EditTextPreference
import androidx.preference.Preference
import androidx.preference.PreferenceFragmentCompat
import com.taobao.meta.avatar.R
import com.taobao.meta.avatar.utils.AppUtils

class MainSettingsFragment : PreferenceFragmentCompat() {
    override fun onCreatePreferences(savedInstanceState: Bundle?, rootKey: String?) {
        setPreferencesFromResource(R.xml.main_settings_prefs, rootKey)
        val checkUpdatePref = findPreference<Preference>("check_update")
        checkUpdatePref?.apply {
            summary = getString(
                R.string.current_version,
                AppUtils.getAppVersionName(requireContext())
            )
        }

        val llmPromptPreference = findPreference<EditTextPreference>("llm_prompt")
        llmPromptPreference?.apply {
            text = MainSettings.getLlmPrompt(requireContext())
            summary = MainSettings.getLlmPrompt(requireContext())
            setOnPreferenceChangeListener { _, newValue ->
                summary = newValue as String
                true
            }
        }
    }
}