// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mls.api.download;

public class DownloadInfo {

    public long totalSize = 0;
    public long savedSize = 0;
    public double progress = 0;

    public int downlodaState = DownloadSate.NOT_START;
    public String currentFile;
    public String progressStage;
    public String errorMessage;

    public static class DownloadSate {
        public static final int NOT_START = 0;

        public static final int DOWNLOADING = 1;

        public static final int COMPLETED = 2;

        public static final int FAILED = 3;

        public static final int PAUSED = 4;


    }
}

