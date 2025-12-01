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
     * Check if user has agreed to the current version of privacy policy
     */
    fun hasUserAgreed(): Boolean {
        val agreedVersion = prefs.getInt(KEY_AGREEMENT_VERSION, 0)
        return prefs.getBoolean(KEY_AGREED, false) && agreedVersion >= CURRENT_POLICY_VERSION
    }
    
    /**
     * Mark that user has agreed to privacy policy
     */
    fun setUserAgreed(agreed: Boolean) {
        prefs.edit().apply {
            putBoolean(KEY_AGREED, agreed)
            if (agreed) {
                putInt(KEY_AGREEMENT_VERSION, CURRENT_POLICY_VERSION)
            }
            apply()
        }
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
        return getUserAgreedVersion() < CURRENT_POLICY_VERSION
    }
    
    /**
     * Clear all privacy policy data (for testing or reset purposes)
     */
    fun clearAgreement() {
        prefs.edit().clear().apply()
    }
}
