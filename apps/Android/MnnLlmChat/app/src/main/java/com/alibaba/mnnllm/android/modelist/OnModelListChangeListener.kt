// Created by mnnchat-issue-killer.
// Copyright (c) 2026 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.modelist

/**
 * Listener for model list changes that affect other UI (e.g. ModelMarketFragment).
 * When a model is deleted from [ModelListFragment], the market tab should refresh
 * so downloaded state stays in sync.
 */
interface OnModelListChangeListener {
    /**
     * Called when the set of downloaded models has changed (e.g. a model was deleted).
     */
    fun onDownloadedModelsChanged()
}
