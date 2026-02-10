// Created by ruoyi.sjd on 2025/3/26.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.taobao.meta.avatar.settings

import android.os.Bundle
import android.widget.Toast
import androidx.preference.EditTextPreference
import androidx.preference.ListPreference
import androidx.preference.Preference
import androidx.preference.PreferenceFragmentCompat
import androidx.preference.SeekBarPreference
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
        
        // TTS Speaker ID 设置（仅英文模式支持，支持动态切换）
        val speakerIdPref = findPreference<ListPreference>("tts_speaker_id")
        speakerIdPref?.apply {
            val isChinese = AppUtils.isChinese()
            // 只有英文模式才启用 speaker_id 设置
            isEnabled = !isChinese
            if (!isChinese) {
                value = MainSettings.getTtsSpeakerId(requireContext())
                setOnPreferenceChangeListener { _, newValue ->
                    val newSpeakerId = newValue as String
                    MainSettings.setTtsSpeakerId(requireContext(), newSpeakerId)
                    // TtsService 会自动监听 SharedPreferences 变化并应用
                    Toast.makeText(requireContext(), 
                        "Voice changed to $newSpeakerId", 
                        Toast.LENGTH_SHORT).show()
                    true
                }
            } else {
                summary = "Only available in English mode"
            }
        }
        
        // TTS Speed 设置（暂时隐藏，不支持设置）
        val speedPref = findPreference<SeekBarPreference>("tts_speed")
        speedPref?.isVisible = false
    }
}