// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited. All rights reserved.
package com.alibaba.mls.api

class HfFileMetadata {
    @JvmField
    var commitHash: String? = null
    @JvmField
    var location: String? = null
    @JvmField
    var etag: String? = null
    @JvmField
    var size: Long = 0
}
