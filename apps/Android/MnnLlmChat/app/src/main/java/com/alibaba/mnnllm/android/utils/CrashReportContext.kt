package com.alibaba.mnnllm.android.utils

import com.google.firebase.crashlytics.FirebaseCrashlytics

/**
 * Centralized Crashlytics context updater so crash reports always include
 * latest UI and model execution state.
 */
object CrashReportContext {
    private const val KEY_CURRENT_ACTIVITY = "current_activity"
    private const val KEY_MODEL_ID = "model_id"
    private const val KEY_SESSION_ID = "session_id"

    fun setCurrentActivity(activityName: String?) {
        updateSafely {
            val value = activityName?.takeIf { it.isNotBlank() } ?: "none"
            setCustomKey(KEY_CURRENT_ACTIVITY, value)
            log("activity=$value")
        }
    }

    fun setCurrentModel(modelId: String?, sessionId: String?) {
        updateSafely {
            val modelValue = modelId?.takeIf { it.isNotBlank() } ?: "none"
            val sessionValue = sessionId?.takeIf { it.isNotBlank() } ?: "none"
            setCustomKey(KEY_MODEL_ID, modelValue)
            setCustomKey(KEY_SESSION_ID, sessionValue)
            log("model=$modelValue session=$sessionValue")
        }
    }

    private inline fun updateSafely(block: FirebaseCrashlytics.() -> Unit) {
        try {
            FirebaseCrashlytics.getInstance().block()
        } catch (_: Throwable) {
            // Do not affect app behavior if crash reporting SDK is unavailable.
        }
    }
}

