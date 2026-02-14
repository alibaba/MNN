// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited. All rights reserved.
package com.alibaba.mls.api

/**
 * Represents a single item in the HuggingFace tree API response.
 * The /api/models/{repo}/tree/{revision} endpoint returns an array of these items.
 */
class HfTreeItem {
    var type: String? = null  // "file" or "directory"
    var oid: String? = null   // object ID (commit SHA)
    var size: Long = 0        // file size in bytes
    var path: String? = null  // relative file path
    var lfs: LfsInfo? = null  // LFS metadata if file is stored in LFS
    
    class LfsInfo {
        var oid: String? = null
        var size: Long = 0
        var pointerSize: Int = 0
    }
    
    /**
     * Get the actual file size, considering LFS files
     */
    fun getActualSize(): Long {
        return lfs?.size ?: size
    }
}
