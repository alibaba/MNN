// Created by ruoyi.sjd on 2025/2/10.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mls.api.ms

class MsRepoInfo {
    var Code: Int = 0
    var Data: ResponseData? = null
    var Message: String? = null
    var Success: Boolean = false

    class ResponseData {
        var Files: List<FileInfo>? = null
        var LatestCommitter:Committer? = null
    }

    class Committer {
        var CreatedAt:Long? = null
    }

    class FileInfo {
        var Name: String? = null
        var Path: String? = null

        var Revision: String? = null
        var Size: Long = 0

        var Sha256: String? = null
        var Type: String? = null
    }
}