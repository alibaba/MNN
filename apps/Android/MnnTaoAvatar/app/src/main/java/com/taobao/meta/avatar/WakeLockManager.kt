// Created by ruoyi.sjd on 2025/3/24.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.taobao.meta.avatar

import android.content.Context
import android.os.PowerManager
import android.util.Log

class WakeLockManager(private val context: Context) {

    private var wakeLock: PowerManager.WakeLock? = null

    /**
     * Acquires a partial wake lock for a specified duration (in milliseconds).
     */
    fun acquireWakeLock(timeout: Long = 10 * 60 * 1000L) { // Default is 10 minutes
        val powerManager = context.getSystemService(Context.POWER_SERVICE) as PowerManager
        wakeLock = powerManager.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "Avatar::WakeLockTag")

        // Check if wakeLock is null or already held to avoid redundant acquisition.
        if (wakeLock?.isHeld == false) {
            wakeLock?.acquire(timeout)
            Log.d("WakeLockManager", "Wake lock acquired for $timeout ms")
        } else {
            Log.d("WakeLockManager", "Wake lock is already held or null")
        }
    }

    /**
     * Releases the wake lock if it is held.
     */
    fun releaseWakeLock() {
        if (wakeLock?.isHeld == true) {
            wakeLock?.release()
            Log.d("WakeLockManager", "Wake lock released")
        } else {
            Log.d("WakeLockManager", "No wake lock held to release")
        }
    }
}