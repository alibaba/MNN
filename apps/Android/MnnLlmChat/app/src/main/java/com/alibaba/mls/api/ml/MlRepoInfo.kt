// Created by ruoyi.sjd on 2025/5/14.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mls.api.ml

data class MlRepoInfo(
    val code: String,
    val msg: String,
    val data: MlRepoData
)

data class MlRepoData(
    val tree: List<FileInfo>,
    val last_commit: LastCommitInfo?,
    val commit_count: Int
)

data class FileInfo(
    val name: String,
    val path: String,
    val type: String,
    val size: Long,
    val is_lfs: Boolean,
    val etag: String,
    val url: String,
    val commit: CommitInfo?,
    val file_scan: FileScanInfo?
)

data class CommitInfo(
    val message: String,
    val commit_sha: String,
    val created: String
)

data class FileScanInfo(
    val status: String,
    val virus: String,
    val sensitive_item: String,
    val moderation_status: String,
    val moderation_result: String
)

data class LastCommitInfo(
    val commit: CommitInfo?,
    val author: AuthorInfo?
)

data class AuthorInfo(
    val name: String,
    val avatar_url: String
)
