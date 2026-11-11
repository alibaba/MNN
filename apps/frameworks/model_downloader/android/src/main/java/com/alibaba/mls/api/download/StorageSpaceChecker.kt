package com.alibaba.mls.api.download

import android.content.Context
import android.os.Build
import android.os.StatFs
import android.os.storage.StorageManager
import android.text.format.Formatter
import android.util.Log
import java.io.File

/**
 * Refuses a download up front when the rest of it cannot fit, instead of failing halfway
 * with a partial model on disk and a low-level write error.
 */
object StorageSpaceChecker {
    private const val TAG = "StorageSpaceChecker"

    /** Headroom for config edits, mmap caches and the system. */
    const val SAFETY_MARGIN_BYTES = 200L * 1024 * 1024

    /** Bytes missing for [remainingBytes] plus the margin; 0 when it fits. */
    fun shortfall(availableBytes: Long, remainingBytes: Long): Long {
        val required = remainingBytes.coerceAtLeast(0) + SAFETY_MARGIN_BYTES
        return (required - availableBytes).coerceAtLeast(0)
    }

    /**
     * A user-facing error when the remaining download does not fit in [dir], or null when
     * it fits or free space cannot be determined (the download then reports any real error).
     */
    fun check(context: Context?, dir: File, remainingBytes: Long): String? {
        if (context == null || remainingBytes <= 0) {
            return null
        }
        val available = availableBytes(context, dir) ?: return null
        if (shortfall(available, remainingBytes) == 0L) {
            return null
        }
        val required = remainingBytes + SAFETY_MARGIN_BYTES
        Log.w(TAG, "Insufficient storage: required=$required available=$available dir=$dir")
        return context.getString(
            R.string.download_insufficient_storage,
            Formatter.formatShortFileSize(context, required),
            Formatter.formatShortFileSize(context, available)
        )
    }

    /**
     * Space the app can use, including cache the system would clear for it (API 26+).
     * The download directory may not exist yet, so the nearest existing ancestor is used.
     */
    fun availableBytes(context: Context, dir: File): Long? {
        var target: File? = dir.absoluteFile
        while (target != null && !target.exists()) {
            target = target.parentFile
        }
        val path = target ?: return null
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            try {
                val storageManager = context.getSystemService(StorageManager::class.java)
                if (storageManager != null) {
                    return storageManager.getAllocatableBytes(storageManager.getUuidForPath(path))
                }
            } catch (e: Exception) {
                Log.w(TAG, "getAllocatableBytes failed for $path, falling back to StatFs", e)
            }
        }
        return try {
            StatFs(path.absolutePath).availableBytes
        } catch (e: Exception) {
            Log.w(TAG, "StatFs failed for $path", e)
            null
        }
    }
}
