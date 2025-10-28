// Created by ruoyi.sjd on 2025/1/13.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.chat.model

class SessionItem(@JvmField val sessionId: String,
                  @JvmField val modelId: String,
                  @JvmField var title: String,
                  @JvmField val lastChatTime: Long = 0L)