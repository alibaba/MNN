package com.alibaba.mnnllm.android.privacy

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.annotation.Config
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class PrivacyPolicyManagerTest {

    private lateinit var manager: PrivacyPolicyManager

    @Before
    fun setUp() {
        manager = PrivacyPolicyManager.getInstance(RuntimeEnvironment.getApplication())
        manager.clearAgreement()
    }

    @Test
    fun `agree should mark choice done and enable crash reporting consent`() {
        manager.setUserConsent(consented = true)

        assertTrue(manager.hasUserMadeChoice())
        assertTrue(manager.isCrashReportingConsented())
    }

    @Test
    fun `disagree should still mark choice done and disable crash reporting consent`() {
        manager.setUserConsent(consented = false)

        assertTrue(manager.hasUserMadeChoice())
        assertFalse(manager.isCrashReportingConsented())
    }

    @Test
    fun `default consent should be enabled before user choice`() {
        assertFalse(manager.hasUserMadeChoice())
        assertTrue(manager.isCrashReportingConsented())
    }
}
