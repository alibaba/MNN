package com.alibaba.mnnllm.api.openai.service

import android.content.Context
import android.content.SharedPreferences
import timber.log.Timber

/**
 * API服务器配置管理类
 * 负责管理服务器的配置参数，包括端口、IP地址、CORS和认证设置
 */
object ApiServerConfig {
    private const val TAG = "ApiServerConfig"
    private const val PREFS_NAME = "api_settings"

    // 配置键名
    private const val KEY_PORT = "port"
    private const val KEY_IP_ADDRESS = "ip_address"
    private const val KEY_CORS_ENABLED = "cors_enabled"
    private const val KEY_CORS_ORIGINS = "cors_origins"
    private const val KEY_AUTH_ENABLED = "auth_enabled"
    private const val KEY_API_KEY = "api_key"
    private const val KEY_CONFIG_INITIALIZED = "config_initialized"

    // 默认配置值
    private const val DEFAULT_PORT = 8080
    private const val DEFAULT_IP_ADDRESS = "127.0.0.1"
    private const val DEFAULT_CORS_ENABLED = false
    private const val DEFAULT_CORS_ORIGINS = ""
    private const val DEFAULT_AUTH_ENABLED = true

    /**
     * 生成随机API Key
     * 8位数字字母符号组合
     */
    private fun generateRandomApiKey(): String {
        val chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*"
        return (1..8)
            .map { chars.random() }
            .joinToString("")
    }

    /**
     * 初始化配置
     * 如果是第一次运行，会保存默认配置到SharedPreferences
     */
    fun initializeConfig(context: Context) {
        val prefs = getPreferences(context)
        val isInitialized = prefs.getBoolean(KEY_CONFIG_INITIALIZED, false)

        if (!isInitialized) {
            Timber.Forest.tag(TAG).i("First time initialization, saving default config")
            saveDefaultConfig(prefs)
        } else {
            Timber.Forest.tag(TAG).i("Config already initialized, loading existing settings")
        }

        logCurrentConfig(context)
    }

    /**
     * 获取服务器端口
     */
    fun getPort(context: Context): Int {
        return getPreferences(context).getInt(KEY_PORT, DEFAULT_PORT)
    }

    /**
     * 获取绑定IP地址
     */
    fun getIpAddress(context: Context): String {
        return getPreferences(context).getString(KEY_IP_ADDRESS, DEFAULT_IP_ADDRESS) ?: DEFAULT_IP_ADDRESS
    }

    /**
     * 获取CORS启用状态
     */
    fun isCorsEnabled(context: Context): Boolean {
        return getPreferences(context).getBoolean(KEY_CORS_ENABLED, DEFAULT_CORS_ENABLED)
    }

    /**
     * 获取CORS允许的来源
     */
    fun getCorsOrigins(context: Context): String {
        return getPreferences(context).getString(KEY_CORS_ORIGINS, DEFAULT_CORS_ORIGINS) ?: DEFAULT_CORS_ORIGINS
    }

    /**
     * 获取认证启用状态
     */
    fun isAuthEnabled(context: Context): Boolean {
        return getPreferences(context).getBoolean(KEY_AUTH_ENABLED, DEFAULT_AUTH_ENABLED)
    }

    /**
     * 获取API密钥
     */
    fun getApiKey(context: Context): String {
        val prefs = getPreferences(context)
        val apiKey = prefs.getString(KEY_API_KEY, "")

        // 如果API Key为空，说明配置可能未初始化，强制初始化
        if (apiKey.isNullOrBlank()) {
            initializeConfig(context)
            return prefs.getString(KEY_API_KEY, "") ?: ""
        }

        return apiKey
    }

    /**
     * 保存配置
     */
    fun saveConfig(
        context: Context,
        port: Int,
        ipAddress: String,
        corsEnabled: Boolean,
        corsOrigins: String,
        authEnabled: Boolean,
        apiKey: String
    ) {
        val prefs = getPreferences(context)
        prefs.edit().apply {
         putInt(KEY_PORT, port)
         putString(KEY_IP_ADDRESS, ipAddress)
         putBoolean(KEY_CORS_ENABLED, corsEnabled)
         putString(KEY_CORS_ORIGINS, corsOrigins)
         putBoolean(KEY_AUTH_ENABLED, authEnabled)
         putString(KEY_API_KEY, apiKey)
         putBoolean(KEY_CONFIG_INITIALIZED, true)
        }.apply()

        Timber.Forest.tag(TAG).i("Config saved: port=$port, ip=$ipAddress, cors=$corsEnabled, auth=$authEnabled")
    }

    /**
     * 重置为默认配置
     */
    fun resetToDefault(context: Context) {
        val prefs = getPreferences(context)
        saveDefaultConfig(prefs)
        Timber.Forest.tag(TAG).i("Config reset to default values")
    }

    /**
     * 获取SharedPreferences实例
     */
    private fun getPreferences(context: Context): SharedPreferences {
        return context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
    }

    /**
     * 保存默认配置
     */
    private fun saveDefaultConfig(prefs: SharedPreferences) {
        val defaultApiKey = generateRandomApiKey()
        prefs.edit().apply {
            putInt(KEY_PORT, DEFAULT_PORT)
            putString(KEY_IP_ADDRESS, DEFAULT_IP_ADDRESS)
            putBoolean(KEY_CORS_ENABLED, DEFAULT_CORS_ENABLED)
            putString(KEY_CORS_ORIGINS, DEFAULT_CORS_ORIGINS)
            putBoolean(KEY_AUTH_ENABLED, DEFAULT_AUTH_ENABLED)
            putString(KEY_API_KEY, defaultApiKey)
            putBoolean(KEY_CONFIG_INITIALIZED, true)
        }.apply()

        Timber.Forest.tag(TAG).i("Generated new API Key: $defaultApiKey")
    }

    /**
     * 记录当前配置到日志
     */
    private fun logCurrentConfig(context: Context) {
        val port = getPort(context)
        val ipAddress = getIpAddress(context)
        val corsEnabled = isCorsEnabled(context)
        val authEnabled = isAuthEnabled(context)

        Timber.Forest.tag(TAG).i("Current config - Port: $port, IP: $ipAddress, CORS: $corsEnabled, Auth: $authEnabled")
    }
}