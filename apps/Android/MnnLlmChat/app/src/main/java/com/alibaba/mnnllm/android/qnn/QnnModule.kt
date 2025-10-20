// Created by ruoyi.sjd on 2025/8/27.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.qnn

import android.content.Context
import android.os.Build
import android.system.Os
import android.util.Log
import java.io.File

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

    /**
     * Check if QNN libraries have already been copied to the app's libs directory
     */
    fun isQnnLibsCopied(context: Context): Boolean {
        val sharedPrefs = context.getSharedPreferences("qnn_libs", Context.MODE_PRIVATE)
        return sharedPrefs.getBoolean("qnn_libs_copied", false)
    }

    /**
     * Mark QNN libraries as copied in SharedPreferences
     */
    fun markQnnLibsCopied(context: Context) {
        val sharedPrefs = context.getSharedPreferences("qnn_libs", Context.MODE_PRIVATE)
        sharedPrefs.edit().putBoolean("qnn_libs_copied", true).apply()
    }

    /**
     * Copy .so files from source directory to target directory
     */
    fun copySoFiles(sourceDir: File, targetDir: File): Boolean {
        try {
            if (!sourceDir.exists() || !sourceDir.isDirectory) {
                log("e", TAG, "Source directory does not exist or is not a directory: ${sourceDir.absolutePath}")
                return false
            }
            
            var copiedCount = 0
            sourceDir.walkTopDown().forEach { file ->
                if (file.isFile && file.extension == "so") {
                    val targetFile = File(targetDir, file.name)
                    try {
                        file.copyTo(targetFile, overwrite = true)
                        copiedCount++
                        log("i", TAG, "Copied ${file.name} to ${targetFile.absolutePath}")
                    } catch (e: Exception) {
                        log("e", TAG, "Failed to copy ${file.name}", e)
                    }
                }
            }
            
            log("i", TAG, "Copied $copiedCount .so files")
            return copiedCount > 0
        } catch (e: Exception) {
            log("e", TAG, "Error copying .so files", e)
            return false
        }
    }

    /**
     * Copy QNN libraries from downloaded directory to the app's libs directory
     */
    fun copyQnnLibs(context: Context, downloadedFile: File): Boolean {
        try {
            log("i", TAG, "Starting QNN libraries copy from ${downloadedFile.absolutePath}")
            
            // Create target libs directory
            val targetLibsDir = File(context.filesDir, "libs")
            if (!targetLibsDir.exists()) {
                targetLibsDir.mkdirs()
            }
            
            // Copy .so files from downloaded directory to target libs directory
            val copied = copySoFiles(downloadedFile, targetLibsDir)
            if (copied) {
                markQnnLibsCopied(context)
                log("i", TAG, "QNN ARM64 libraries successfully copied to ${targetLibsDir.absolutePath}")
                return true
            } else {
                log("e", TAG, "Failed to copy QNN libraries")
                return false
            }
            
        } catch (e: Exception) {
            log("e", TAG, "Error copying QNN libs", e)
            return false
        }
    }
}