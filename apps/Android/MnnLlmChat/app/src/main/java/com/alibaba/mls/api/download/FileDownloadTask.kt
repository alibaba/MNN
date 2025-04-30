// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mls.api.download

import com.alibaba.mls.api.HfFileMetadata
import java.io.File

class FileDownloadTask {
    @JvmField
    var relativePath: String? = null
    @JvmField
    var hfFileMetadata: HfFileMetadata? = null
    @JvmField
    var blobPath: File? = null
    @JvmField
    var blobPathIncomplete: File? = null
    @JvmField
    var pointerPath: File? = null

    @JvmField
    var downloadedSize: Long = 0
}
