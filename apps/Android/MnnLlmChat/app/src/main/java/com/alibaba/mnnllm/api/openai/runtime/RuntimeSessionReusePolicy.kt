package com.alibaba.mnnllm.api.openai.runtime

internal object RuntimeSessionReusePolicy {
    fun shouldReuse(
        forceReload: Boolean,
        activeModelId: String?,
        requestedModelId: String,
        isSessionLoaded: Boolean,
        activeUseAppConfig: Boolean,
        requestedUseAppConfig: Boolean
    ): Boolean {
        if (forceReload) return false
        if (activeModelId != requestedModelId) return false
        if (activeUseAppConfig != requestedUseAppConfig) return false
        return isSessionLoaded
    }

    /**
     * A reusable session bound to a different chat conversation must be rebound before use,
     * otherwise the previous conversation's history, KV cache and session id leak into the
     * new one. Callers without a session id (API service) keep whatever binding is active.
     */
    fun needsRebind(requestedSessionId: String?, activeSessionId: String?): Boolean {
        return requestedSessionId != null && requestedSessionId != activeSessionId
    }

    fun shouldExposeActiveSession(isSessionLoaded: Boolean): Boolean {
        return isSessionLoaded
    }
}
