// Created by ruoyi.sjd on 2025/1/13.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.utils

import android.content.Context
import android.preference.PreferenceManager
import com.alibaba.mnnllm.android.utils.DeviceUtils.isChinese

object PreferenceUtils {
    const val TAG: String = "PreferenceUtils"
    const val KEY_SHOW_PERFORMACE_METRICS: String = "SHOW_PERFORMACE_METRICS"

    const val KEY_USE_MODELSCOPE_DOWNLOAD: String = "USE_MODELSCOPE_DOWNLOAD"

    const val KEY_LIST_FILTER_ONLY_DOWNLOADED: String = "LIST_FILTER_ONLY_DOWNLOADED"

    const val KEY_DIFFUSION_MEMORY_MODE: String = "diffusion_memory_mode"

    // Pinned models management
    const val KEY_PINNED_MODELS: String = "PINNED_MODELS"

    // Timber logging configuration
    const val KEY_TIMBER_LOGGING_ENABLED: String = "TIMBER_LOGGING_ENABLED"

    fun setBoolean(context: Context?, key: String?, value: Boolean) {
        PreferenceManager.getDefaultSharedPreferences(context).edit().putBoolean(key, value).apply()
    }

    fun setLong(context: Context?, key: String?, value: Long) {
        PreferenceManager.getDefaultSharedPreferences(context).edit().putLong(key, value).apply()
    }

    fun getLong(context: Context?, key: String?, defaultValue: Long): Long {
        return PreferenceManager.getDefaultSharedPreferences(context).getLong(key, defaultValue)
    }

    fun getBoolean(context: Context?, key: String?, defaultValue: Boolean): Boolean {
        return PreferenceManager.getDefaultSharedPreferences(context).getBoolean(key, defaultValue)
    }

    fun setString(context: Context?, key: String?, value: String?) {
        PreferenceManager.getDefaultSharedPreferences(context).edit().putString(key, value).apply()
    }

    fun getString(context: Context?, key: String?, defaultValue: String?): String? {
        return PreferenceManager.getDefaultSharedPreferences(context).getString(key, defaultValue)
    }

    fun isUseModelsScopeDownload(context: Context?): Boolean {
        return getBoolean(context, KEY_USE_MODELSCOPE_DOWNLOAD, true)
    }

    // Timber logging configuration methods
    fun isTimberLoggingEnabled(context: Context?): Boolean {
        return getBoolean(context, KEY_TIMBER_LOGGING_ENABLED, true) // Default to enabled
    }

    fun setTimberLoggingEnabled(context: Context?, enabled: Boolean) {
        setBoolean(context, KEY_TIMBER_LOGGING_ENABLED, enabled)
    }

    fun isUseModelsScopeDownloadWithDefault(context: Context?): Boolean {
        val defaultValue = isChinese
        if (!PreferenceManager.getDefaultSharedPreferences(context).all.containsKey(
                KEY_USE_MODELSCOPE_DOWNLOAD
            )
        ) {
            setUseModelsScopeDownload(context, defaultValue)
            return defaultValue
        }
        return getBoolean(context, KEY_USE_MODELSCOPE_DOWNLOAD, defaultValue)
    }

    fun setUseModelsScopeDownload(context: Context?, value: Boolean) {
        setBoolean(context, KEY_USE_MODELSCOPE_DOWNLOAD, value)
    }

    @JvmStatic
    fun setFilterDownloaded(context: Context?, filterDownloaded: Boolean) {
        setBoolean(context, KEY_LIST_FILTER_ONLY_DOWNLOADED, filterDownloaded)
    }

    @JvmStatic
    fun isFilterDownloaded(context: Context?): Boolean {
        return getBoolean(context, KEY_LIST_FILTER_ONLY_DOWNLOADED, false)
    }

    // Pinned models management methods
    fun getPinnedModels(context: Context?): Set<String> {
        val pinnedModelsString = getString(context, KEY_PINNED_MODELS, "")
        return if (pinnedModelsString.isNullOrEmpty()) {
            emptySet()
        } else {
            pinnedModelsString.split(",").toSet()
        }
    }

    fun setPinnedModels(context: Context?, pinnedModels: Set<String>) {
        val pinnedModelsString = pinnedModels.joinToString(",")
        setString(context, KEY_PINNED_MODELS, pinnedModelsString)
    }

    fun pinModel(context: Context?, modelId: String) {
        val pinnedModels = getPinnedModels(context).toMutableSet()
        pinnedModels.add(modelId)
        setPinnedModels(context, pinnedModels)
    }

    fun unpinModel(context: Context?, modelId: String) {
        val pinnedModels = getPinnedModels(context).toMutableSet()
        pinnedModels.remove(modelId)
        setPinnedModels(context, pinnedModels)
    }

    fun isModelPinned(context: Context?, modelId: String): Boolean {
        return getPinnedModels(context).contains(modelId)
    }
}
