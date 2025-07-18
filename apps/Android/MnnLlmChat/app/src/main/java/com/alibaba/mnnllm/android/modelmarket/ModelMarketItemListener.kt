package com.alibaba.mnnllm.android.modelmarket

import android.view.View

interface ModelMarketItemListener {
    fun onDownloadOrResumeClicked(item: ModelMarketItemWrapper)
    fun onPauseClicked(item: ModelMarketItemWrapper)
    fun onActionClicked(item: ModelMarketItemWrapper)
    fun onDeleteClicked(item: ModelMarketItemWrapper)
    fun onUpdateClicked(item: ModelMarketItemWrapper)
    fun onDefaultVoiceModelChanged(item: ModelMarketItemWrapper)
} 