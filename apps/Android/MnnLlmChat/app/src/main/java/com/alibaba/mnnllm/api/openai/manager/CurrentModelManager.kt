package com.alibaba.mnnllm.api.openai.manager

/**
 * Current model manager
 * Used to store and access the currently active model ID in the API service
 */
object CurrentModelManager {
    private var currentModelId: String? = null
    
    /**
     * Set current model ID
     */
    fun setCurrentModelId(modelId: String?) {
        currentModelId = modelId
    }
    
    /**
     * Get current model ID
     */
    fun getCurrentModelId(): String? = currentModelId
    
    /**
     * Clear current model ID
     */
    fun clearCurrentModelId() {
        currentModelId = null
    }
}
