package com.alibaba.mnnllm.android.utils

import android.content.Context
import android.os.Bundle
import com.alibaba.mnnllm.android.BuildConfig
import com.alibaba.mnnllm.android.privacy.PrivacyPolicyManager
import com.google.firebase.analytics.FirebaseAnalytics

object AnalyticsTracker {
    private const val EVENT_APP_OPEN = "app_open"
    private const val EVENT_PAGE_VIEW = "page_view"
    private const val PARAM_PAGE_NAME = "page_name"

    fun logAppOpen(context: Context) {
        logEvent(context, EVENT_APP_OPEN, null)
    }

    fun logPageView(context: Context, pageName: String) {
        val params = Bundle().apply {
            putString(PARAM_PAGE_NAME, pageName)
        }
        logEvent(context, EVENT_PAGE_VIEW, params)
    }

    private fun logEvent(context: Context, eventName: String, params: Bundle?) {
        if (!BuildConfig.ENABLE_FIREBASE) return
        if (!PrivacyPolicyManager.getInstance(context).isCrashReportingConsented()) return
        try {
            FirebaseAnalytics.getInstance(context).logEvent(eventName, params)
        } catch (_: Throwable) {
            // Never affect app flow due to analytics SDK state.
        }
    }
}
