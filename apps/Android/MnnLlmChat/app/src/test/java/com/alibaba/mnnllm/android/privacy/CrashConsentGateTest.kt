package com.alibaba.mnnllm.android.privacy

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class CrashConsentGateTest {

    @Test
    fun `shows dialog when firebase enabled and user has not made choice`() {
        assertTrue(CrashConsentGate.shouldShowConsentDialog(firebaseEnabled = true, hasUserMadeChoice = false))
    }

    @Test
    fun `does not show dialog when firebase disabled`() {
        assertFalse(CrashConsentGate.shouldShowConsentDialog(firebaseEnabled = false, hasUserMadeChoice = false))
    }

    @Test
    fun `does not show dialog when user already made choice`() {
        assertFalse(CrashConsentGate.shouldShowConsentDialog(firebaseEnabled = true, hasUserMadeChoice = true))
    }
}
