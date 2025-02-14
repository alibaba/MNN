// Created by ruoyi.sjd on 2025/2/13.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.chat;

import android.app.DownloadManager;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;

import com.alibaba.mnnllm.android.update.UpdateChecker;

public class DownloadReciver extends BroadcastReceiver {
    @Override
    public void onReceive(Context context, Intent intent) {
        long completedDownloadId = intent.getLongExtra(DownloadManager.EXTRA_DOWNLOAD_ID, -1);
        UpdateChecker.installApk(context, completedDownloadId);
    }
}
