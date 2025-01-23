// Created by ruoyi.sjd on 2025/1/13.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.modelist;

public class ModelListContract {
    public interface View {
        void onListAvailable();
        void onLoading();
        void onListLoadError(String error);

        ModelListAdapter getAdapter();

        void runModel(String absolutePath, String modelName);
    }
}
