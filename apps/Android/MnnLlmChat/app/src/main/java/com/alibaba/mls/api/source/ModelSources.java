// Created by ruoyi.sjd on 2025/2/11.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mls.api.source;

import com.alibaba.mls.api.ApplicationUtils;
import com.alibaba.mnnllm.android.utils.PreferenceUtils;

public class ModelSources {

    public enum ModelSourceType {
        MODEL_SCOPE,
        HUGGING_FACE,
        LOCAL
    }

    public ModelSourceType getRemoteSourceType() {
        return PreferenceUtils.isUseModelsScopeDownload(ApplicationUtils.get()) ?
                ModelSourceType.MODEL_SCOPE : ModelSourceType.HUGGING_FACE;
    }

    private static final class InstanceHolder {
        static final ModelSources instance = new ModelSources();
    }

    private static ModelSourceConfig mockConfig;

    public static ModelSources get() {
        return InstanceHolder.instance;
    }

    public ModelSourceConfig getConfig() {
        if (mockConfig == null) {
            mockConfig = ModelSourceConfig.createMockSourceConfig();
        }
        return mockConfig;
    }
}
