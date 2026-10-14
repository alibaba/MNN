// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.hexagon

import android.content.Context
import android.os.Build
import android.system.Os
import android.util.Log
import java.io.File

/**
 * Runtime setup for the MNN Hexagon (HTP/cDSP) backend.
 *
 * The Hexagon backend is a regular MNN backend (forward type "hexagon"), so it is selected
 * per-model through the same backend_type path as cpu/opencl. It is mutually exclusive with the
 * QNN model path: QNN models are handled separately in ChatRouter and never reach here.
 *
 * libMNN.so dlopen()s the FastRPC stub (libMNN_htpops.so) by soname, and FastRPC in turn loads the
 * DSP skeleton (libMNN_htpops_skelV*.so). Both are packaged into the app native library dir, so we
 * only need to point ADSP_LIBRARY_PATH / LD_LIBRARY_PATH at it before a DSP session is opened.
 */
object HexagonModule {
    private const val TAG = "HexagonModule"
    private const val STUB_LIB = "libMNN_htpops.so"

    @Volatile
    private var prepared = false

    // SOC_MODEL -> Hexagon arch. The HTP backend needs FP16 HMX, i.e. Hexagon v73+ (8 Gen2+).
    private val sSupportedSoc = hashMapOf<String, String>().apply {
        put("SM8850", "V81")
        put("SM8750", "V79")
        put("SM8650", "V75")
        put("SM8550", "V73")
    }

    private fun getSocModel(): String? {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            Build.SOC_MODEL
        } else {
            null
        }
    }

    private fun nativeLibDir(context: Context): String = context.applicationInfo.nativeLibraryDir

    /**
     * Whether the Hexagon backend can be offered on this device: a supported SoC and the FastRPC
     * stub actually packaged into the APK.
     */
    fun deviceSupported(context: Context): Boolean {
        val soc = getSocModel()
        if (soc == null || !sSupportedSoc.containsKey(soc)) {
            return false
        }
        val stub = File(nativeLibDir(context), STUB_LIB)
        if (!stub.exists()) {
            Log.w(TAG, "SoC $soc supported but $STUB_LIB not packaged, Hexagon disabled")
            return false
        }
        return true
    }

    /**
     * Point FastRPC at the packaged stub/skel. Safe to call more than once; only takes effect the
     * first time. Must run before the model that uses the hexagon backend is loaded.
     */
    fun prepare(context: Context): Boolean {
        if (prepared) {
            return true
        }
        val path = nativeLibDir(context)
        return try {
            Os.setenv("ADSP_LIBRARY_PATH", path, true)
            Os.setenv("LD_LIBRARY_PATH", path, true)
            prepared = true
            Log.i(TAG, "Hexagon lib path ready: $path (soc=${getSocModel()})")
            true
        } catch (e: Throwable) {
            Log.e(TAG, "Hexagon prepare setenv failed", e)
            false
        }
    }
}
