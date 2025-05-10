// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mls.api.download

import com.alibaba.mls.api.HfFileMetadata
import java.io.File

class FileDownloadTask {
    var etag: String? = null
    var relativePath: String? = null
    var fileMetadata: HfFileMetadata? = null
    var blobPath: File? = null
    var blobPathIncomplete: File? = null
    var pointerPath: File? = null
    var downloadedSize: Long = 0
}
