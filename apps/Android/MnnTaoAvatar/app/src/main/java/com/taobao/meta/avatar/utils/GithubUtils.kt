// Created by ruoyi.sjd on 2025/2/6.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.utils

import android.content.Context
import android.content.Intent
import android.net.Uri

object GithubUtils {

    private const val repoGithubUrl = "https://github.com/alibaba/MNN"

    fun openInBrowser(context: Context, url: String?) {
        val intent = Intent(Intent.ACTION_VIEW, Uri.parse(url))
        context.startActivity(intent)
    }

    fun starProject(context: Context) {
        openInBrowser(context, repoGithubUrl)
    }

    fun reportIssue(context: Context) {
        openInBrowser(context, repoGithubUrl + "/issues")
    }
}