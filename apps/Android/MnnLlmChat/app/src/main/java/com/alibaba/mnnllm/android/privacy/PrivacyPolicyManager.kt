// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.privacy

import android.content.Context
import android.content.SharedPreferences

class PrivacyPolicyManager private constructor(context: Context) {
    
    private val prefs: SharedPreferences = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
    
    companion object {
        private const val PREFS_NAME = "privacy_policy_prefs"
        private const val KEY_AGREED = "privacy_policy_agreed"
        private const val KEY_CRASH_REPORTING_CONSENT = "crash_reporting_consent"
        private const val KEY_AGREEMENT_VERSION = "privacy_policy_version"
        
        // Current version of privacy policy - increment this when policy changes
        private const val CURRENT_POLICY_VERSION = 1
        
        @Volatile
        private var INSTANCE: PrivacyPolicyManager? = null
        
        fun getInstance(context: Context): PrivacyPolicyManager {
            return INSTANCE ?: synchronized(this) {
                INSTANCE ?: PrivacyPolicyManager(context.applicationContext).also { INSTANCE = it }
            }
        }
    }
    
    /**
     * Check if user has made a privacy choice for current policy version.
     */
    fun hasUserMadeChoice(): Boolean {
        val choiceVersion = prefs.getInt(KEY_AGREEMENT_VERSION, 0)
        return choiceVersion >= CURRENT_POLICY_VERSION
    }

    /**
     * Backward-compatible API: agreed means choice made and consented to reporting.
     */
    fun hasUserAgreed(): Boolean {
        return hasUserMadeChoice() && isCrashReportingConsented()
    }
    
    /**
     * Persist user decision for crash data reporting.
     */
    fun setUserConsent(consented: Boolean) {
        prefs.edit().apply {
            putBoolean(KEY_AGREED, consented)
            putBoolean(KEY_CRASH_REPORTING_CONSENT, consented)
            putInt(KEY_AGREEMENT_VERSION, CURRENT_POLICY_VERSION)
            apply()
        }
    }

    /**
     * Backward-compatible API: treat old agreement as reporting consent.
     */
    fun setUserAgreed(agreed: Boolean) {
        setUserConsent(agreed)
    }

    fun isCrashReportingConsented(): Boolean {
        if (!hasUserMadeChoice()) {
            return true
        }
        return prefs.getBoolean(KEY_CRASH_REPORTING_CONSENT, prefs.getBoolean(KEY_AGREED, false))
    }
    
    /**
     * Get the current privacy policy version
     */
    fun getCurrentPolicyVersion(): Int {
        return CURRENT_POLICY_VERSION
    }
    
    /**
     * Get the version user agreed to
     */
    fun getUserAgreedVersion(): Int {
        return prefs.getInt(KEY_AGREEMENT_VERSION, 0)
    }
    
    /**
     * Check if user needs to re-agree due to policy version change
     */
    fun needsReAgreement(): Boolean {
        return !hasUserMadeChoice()
    }
    
    /**
     * Clear all privacy policy data (for testing or reset purposes)
     */
    fun clearAgreement() {
        prefs.edit().clear().apply()
    }
}
