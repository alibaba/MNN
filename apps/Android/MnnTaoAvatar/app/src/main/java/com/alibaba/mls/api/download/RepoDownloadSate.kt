// Created by ruoyi.sjd on 2025/4/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mls.api.download

//download state for a single file

object RepoDownloadSate {

    const val NOT_START: Int = 0

    const val DOWNLOADING: Int = 1

    const val COMPLETED: Int = 2

    const val FAILED: Int = 3

    const val PAUSED: Int = 4

    const val PREPARING: Int = 5
}