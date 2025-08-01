// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mls.api.download

import com.alibaba.mnnllm.android.R

object DownloadState {
    const val NOT_START: Int = 0

    const val DOWNLOADING: Int = 1

    const val COMPLETED: Int = 2

    const val FAILED: Int = 3

    const val PAUSED: Int = 4

    val downloadStateList = listOf(
        COMPLETED,
        DOWNLOADING,
        FAILED,
        PAUSED,
        NOT_START
    )

    fun toStringRes(state: Int): Int {
        return when (state) {
            COMPLETED -> R.string.download_state_completed
            NOT_START -> R.string.download_state_not_start
            DOWNLOADING -> R.string.download_state_downloading
            FAILED -> R.string.download_state_failed
            PAUSED -> R.string.download_state_paused
            else -> R.string.download_state_failed
        }
    }
} 