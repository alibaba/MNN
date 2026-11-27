package com.alibaba.mnnllm.api.openai.runtime

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class RuntimeSessionReusePolicyTest {

    @Test
    fun shouldReuse_returnsFalse_whenCachedSessionIsNotLoaded() {
        assertFalse(
            RuntimeSessionReusePolicy.shouldReuse(
                forceReload = false,
                activeModelId = "same-model",
                requestedModelId = "same-model",
                isSessionLoaded = false,
                activeUseAppConfig = false,
                requestedUseAppConfig = false
            )
        )
    }

    @Test
    fun shouldReuse_returnsTrue_whenCachedSessionIsLoadedForSameModel() {
        assertTrue(
            RuntimeSessionReusePolicy.shouldReuse(
                forceReload = false,
                activeModelId = "same-model",
                requestedModelId = "same-model",
                isSessionLoaded = true,
                activeUseAppConfig = true,
                requestedUseAppConfig = true
            )
        )
    }

    @Test
    fun shouldReuse_returnsFalse_whenAppConfigModeChanges() {
        assertFalse(
            RuntimeSessionReusePolicy.shouldReuse(
                forceReload = false,
                activeModelId = "same-model",
                requestedModelId = "same-model",
                isSessionLoaded = true,
                activeUseAppConfig = false,
                requestedUseAppConfig = true
            )
        )
    }

    @Test
    fun shouldExposeActiveSession_returnsFalse_whenSessionIsNotLoaded() {
        assertFalse(RuntimeSessionReusePolicy.shouldExposeActiveSession(isSessionLoaded = false))
    }

    @Test
    fun needsRebind_returnsTrue_whenRequestedSessionDiffers() {
        assertTrue(RuntimeSessionReusePolicy.needsRebind("session-b", "session-a"))
    }

    @Test
    fun needsRebind_returnsFalse_whenSameSession() {
        assertFalse(RuntimeSessionReusePolicy.needsRebind("session-a", "session-a"))
    }

    @Test
    fun needsRebind_returnsFalse_whenCallerHasNoSessionId() {
        // API-service style callers keep whatever conversation binding is active.
        assertFalse(RuntimeSessionReusePolicy.needsRebind(null, "session-a"))
    }

    @Test
    fun needsRebind_returnsTrue_whenActiveSessionHasNoIdYet() {
        assertTrue(RuntimeSessionReusePolicy.needsRebind("session-a", null))
    }
}
