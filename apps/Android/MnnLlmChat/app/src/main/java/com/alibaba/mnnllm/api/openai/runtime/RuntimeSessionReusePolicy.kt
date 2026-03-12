package com.alibaba.mnnllm.api.openai.runtime

internal object RuntimeSessionReusePolicy {
    fun shouldReuse(
        forceReload: Boolean,
        activeModelId: String?,
        requestedModelId: String,
        isSessionLoaded: Boolean
    ): Boolean {
        if (forceReload) return false
        if (activeModelId != requestedModelId) return false
        return isSessionLoaded
    }

    fun shouldExposeActiveSession(isSessionLoaded: Boolean): Boolean {
        return isSessionLoaded
    }
}
