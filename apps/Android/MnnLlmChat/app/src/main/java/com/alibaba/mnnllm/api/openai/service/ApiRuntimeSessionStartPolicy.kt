package com.alibaba.mnnllm.api.openai.service

internal object ApiRuntimeSessionStartPolicy {
    fun shouldReuseLoadedSession(
        requestedModelId: String?,
        activeModelId: String?,
        hasLoadedActiveSession: Boolean
    ): Boolean {
        if (!hasLoadedActiveSession) {
            return false
        }
        return requestedModelId.isNullOrBlank() || requestedModelId == activeModelId
    }
}
