// Created by ruoyi.sjd on 2025/8/27.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.qnn

import android.content.Context
import android.os.Build
import android.system.Os
import android.util.Log

object QnnModule {
    private const val TAG = "QnnModule"

    @Volatile
    private var qnnInit = false

    //SOC_MODEL -> Hexagon Arch mapping table
    //Based on https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/overview.html#supported-snapdragon-devices
    private val sQnnConfigMap = hashMapOf<String, String>().apply {
        put("SM8750", "V79")
        put("SM8650", "V75")
        put("SM8550", "V73")
        put("SM8475", "V69")
        put("SM8450", "V69")
        put("SM8350P", "V69")
        put("SM8350", "V68")
    }

    private fun loadQnnLibrary(libraryName: String, nativeLibPath: String): Boolean {
        return try {
            val fullLibPath = "${nativeLibPath}lib${libraryName}.so"
            System.load(fullLibPath)
            log("i", TAG, "loadLibrary $libraryName from $fullLibPath Success")
            true
        } catch (e: UnsatisfiedLinkError) {
            log("e", TAG, "loadLibrary $libraryName from $nativeLibPath Fail: ${e.message}")
            false
        } catch (e: Exception) {
            log("e", TAG, "loadLibrary $libraryName from $nativeLibPath Exception: ${e.message}")
            false
        }
    }

    private fun log(level: String, tag: String, content: String, throwable: Throwable? = null) {
        Log.i(tag, content, throwable)
//        when (level) {
//            "e" -> Log.e(tag, content, throwable)
//            "w" -> Log.w(tag, content, throwable)
//            "i" -> Log.i(tag, content, throwable)
//            "d" -> Log.d(tag, content, throwable)
//            else -> Log.v(tag, content, throwable)
//        }
    }

    private fun getSocModel(): String? {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            Build.SOC_MODEL
        } else {
            null
        }
    }

    @Synchronized
    fun loadQnnLibs(context: Context): Boolean {
        if (qnnInit) {
            return true
        }
        qnnInit = true

        val configMap = sQnnConfigMap
        val socModel = getSocModel()

        if (socModel == null) {
            log("e", TAG, "qnn not supported for socId null")
            return false
        }

        val qnnVersion = configMap[socModel]
        log("w", TAG, "load qnn libs socId: $socModel qnnVersion: $qnnVersion")

        if (qnnVersion == null) {
            log("e", TAG, "qnn not supported for socId: ${getSocModel()}")
            return false
        }

        val nativeLibPath = context.filesDir.absolutePath + "/libs/"
        try {
            log("i", TAG, "loadQnnLibs nativeLibPath: $nativeLibPath")
            Os.setenv("ADSP_LIBRARY_PATH", nativeLibPath, true)
            Os.setenv("LD_LIBRARY_PATH", nativeLibPath, true)
        } catch (e: Throwable) {
            log("e", TAG, "loadQnnLibs setenv failed", e)
            return false
        }

        // Load QnnHtp library
        if (!loadQnnLibrary("QnnHtp", nativeLibPath)) {
            log("e", TAG, "Load libQnnHtp.so failed.")
            return false
        } else {
            log("i", TAG, "Load libQnnHtp.so Success")
        }

        // Load QnnSystem library
        if (!loadQnnLibrary("QnnSystem", nativeLibPath)) {
            log("e", TAG, "Load libQnnSystem.so failed.")
            return false
        } else {
            log("i", TAG, "Load libQnnSystem.so Success")
        }

        // Load version-specific libraries
        loadQnnLibrary("QnnHtp${qnnVersion}Skel", nativeLibPath)
        loadQnnLibrary("QnnHtp${qnnVersion}Stub", nativeLibPath)

        return true
    }
}