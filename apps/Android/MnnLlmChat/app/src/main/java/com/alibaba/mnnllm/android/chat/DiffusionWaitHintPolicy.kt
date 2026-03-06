package com.alibaba.mnnllm.android.chat

import com.alibaba.mnnllm.android.model.ModelTypeUtils

object DiffusionWaitHintPolicy {
    fun shouldShowWaitHint(modelName: String): Boolean {
        return ModelTypeUtils.isDiffusionModel(modelName)
    }
}
