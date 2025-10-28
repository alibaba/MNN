// Created by ruoyi.sjd on 2025/8/27.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.qnn

import android.content.Context
import android.os.Build
import android.system.Os
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

object QnnModule {
    private const val TAG = "QnnModule"
    

    @Volatile
    private var qnnInit = false

    //SOC_MODEL -> Hexagon Arch mapping table
    //Based on https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/overview.html#supported-snapdragon-devices
    private val sQnnLibMap = hashMapOf<String, String>().apply {
        put("SM8750", "V79")
        put("SM8650", "V75")
        put("SM8550", "V73")
        put("SM8475", "V69")
        put("SM8450", "V69")
        put("SM8350P", "V68")
        put("SM8350", "V68")
    }

    private val sQnnModelNameMap = hashMapOf<String, String>().apply {
        put("SM8750", "69_v79")
        put("SM8650", "57_v75")
        put("SM8550", "43_v73")
        put("SM8475", "42_v69")
        put("SM8450", "36_v69")
        put("SM8350P", "30_v68")
        put("SM8350", "30_v68")
    }

    fun modelMiddleName(): String? {
        return sQnnModelNameMap[getSocModel()]
    }

    private fun loadQnnLibrary(libraryName: String, nativeLibPath: String): Boolean {
        return try {
            val fullLibPath = "${nativeLibPath}${File.separator}lib${libraryName}.so"
            System.load(fullLibPath)
            Log.i(TAG, "loadLibrary $libraryName from $fullLibPath Success")
            true
        } catch (e: UnsatisfiedLinkError) {
            Log.e(TAG, "loadLibrary $libraryName from $nativeLibPath Fail: ${e.message}")
            false
        } catch (e: Exception) {
            Log.e(TAG, "loadLibrary $libraryName from $nativeLibPath Exception: ${e.message}")
            false
        }
    }


    private fun getSocModel(): String? {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            Build.SOC_MODEL
        } else {
            null
        }
    }

    /**
     * Get QNN library version for current SOC model
     * @return QNN library version string or null if not supported
     */
    private fun getQnnLibVersion(): String? {
        val socModel = getSocModel()
        return if (socModel != null) {
            sQnnLibMap[socModel]
        } else {
            null
        }
    }

    fun deviceSupported(): Boolean {
        return getQnnLibVersion() != null
    }

    /**
     * Load QNN libraries from the stored download path
     */
    suspend fun loadQnnLibs(context: Context): Boolean = withContext(Dispatchers.IO) {
        if (qnnInit) {
            return@withContext true
        }
        val qnnVersion = getQnnLibVersion()
        if (qnnVersion == null) {
            Log.e(TAG, "qnn not supported for socId: ${getSocModel()}")
            return@withContext false
        }
        
        // Use stored download path
        val nativeLibPath = getQnnLibsPath(context) ?: run {
            Log.e(TAG, "QNN libs path not found")
            return@withContext false
        }

        try {
            Log.i(TAG, "loadQnnLibs nativeLibPath: $nativeLibPath")
            Os.setenv("ADSP_LIBRARY_PATH", nativeLibPath, true)
            Os.setenv("LD_LIBRARY_PATH", nativeLibPath, true)
        } catch (e: Throwable) {
            Log.e(TAG, "loadQnnLibs setenv failed", e)
            return@withContext false
        }

        // Load QnnHtp library
        if (!loadQnnLibrary("QnnHtp", nativeLibPath)) {
            Log.e(TAG, "Load libQnnHtp.so failed.")
            return@withContext false
        } else {
            Log.i(TAG, "Load libQnnHtp.so Success")
        }

        // Load QnnSystem library
        if (!loadQnnLibrary("QnnSystem", nativeLibPath)) {
            Log.e(TAG, "Load libQnnSystem.so failed.")
            return@withContext false
        } else {
            Log.i(TAG, "Load libQnnSystem.so Success")
        }

        // Load version-specific libraries
        loadQnnLibrary("QnnHtp${qnnVersion}Skel", nativeLibPath)
        loadQnnLibrary("QnnHtp${qnnVersion}Stub", nativeLibPath)

        qnnInit = true
        return@withContext true
    }

    /**
     * Check if QNN libraries have already been downloaded
     */
    suspend fun isQnnLibsDownloaded(context: Context): Boolean = withContext(Dispatchers.IO) {
        val sharedPrefs = context.getSharedPreferences("qnn_libs", Context.MODE_PRIVATE)
        sharedPrefs.getBoolean("qnn_libs_downloaded", false)
    }

    /**
     * Mark QNN libraries as downloaded in SharedPreferences
     */
    suspend fun markQnnLibsDownloaded(context: Context, downloadPath: String) = withContext(Dispatchers.IO) {
        val sharedPrefs = context.getSharedPreferences("qnn_libs", Context.MODE_PRIVATE)
        sharedPrefs.edit()
            .putBoolean("qnn_libs_downloaded", true)
            .putString("qnn_libs_path", downloadPath)
            .apply()
    }
    
    /**
     * Get the stored QNN libraries download path
     */
    suspend fun getQnnLibsPath(context: Context): String? = withContext(Dispatchers.IO) {
        val sharedPrefs = context.getSharedPreferences("qnn_libs", Context.MODE_PRIVATE)
        sharedPrefs.getString("qnn_libs_path", null)
    }

}