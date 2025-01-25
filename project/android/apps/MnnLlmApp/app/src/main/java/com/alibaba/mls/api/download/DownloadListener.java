// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mls.api.download;

public interface DownloadListener {
    void onDownloadStart(String modelId);

    void onDownloadFailed(String modelId, Exception hfApiException);

    void onDownloadProgress(String modelId, DownloadInfo progress);
    void onDownloadFinished(String modelId, String path);
    void onDownloadPaused(String modelId);
    void onDownloadFileRemoved(String modelId);
}