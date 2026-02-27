package com.alibaba.mnnllm.android.privacy

object CrashConsentGate {
    fun shouldShowConsentDialog(firebaseEnabled: Boolean, hasUserMadeChoice: Boolean): Boolean {
        return firebaseEnabled && !hasUserMadeChoice
    }
}
