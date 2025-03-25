// Created by ruoyi.sjd on 2025/2/6.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.utils;

import android.content.Context;
import android.content.Intent;
import android.net.Uri;

public class GithubUtils {

    private static  final String repoGithubUrl = "https://github.com/alibaba/MNN";

    public static void openInBrowser(Context context, String url) {
        Intent intent = new Intent(Intent.ACTION_VIEW, Uri.parse(url));
        context.startActivity(intent);
    }

    public static void starProject(Context context) {
        openInBrowser(context, repoGithubUrl);
    }

    public static void reportIssue(Context context) {
        openInBrowser(context, repoGithubUrl + "/issues");
    }
}
