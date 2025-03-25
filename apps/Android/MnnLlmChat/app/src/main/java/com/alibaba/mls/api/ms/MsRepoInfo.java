// Created by ruoyi.sjd on 2025/2/10.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mls.api.ms;

import java.util.List;


public class MsRepoInfo {
    public int Code;
    public Data Data;
    public String Message;
    public boolean Success;

    public static class Data {
        public List<FileInfo> Files;
    }

    public static class FileInfo {
        public String Name;
        public String Path;

        public String Revision;
        public long Size;

        public String Sha256;
    }
}