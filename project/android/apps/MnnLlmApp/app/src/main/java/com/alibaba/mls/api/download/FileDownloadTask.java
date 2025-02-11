// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mls.api.download;

import com.alibaba.mls.api.HfFileMetadata;

import java.io.File;


public class FileDownloadTask {
    public String relativePath;
    HfFileMetadata hfFileMetadata;
    File blobPath;
    File blobPathIncomplete;
    File pointerPath;

    long resumeSize;

}
