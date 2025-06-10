// Created by ruoyi.sjd on 2025/2/10.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mls.api.ms;

import retrofit2.Call;
import retrofit2.http.GET;
import retrofit2.http.Path;

public interface MsApiService {
    @GET("api/v1/models/{modelGroup}/{modelPath}/repo/files?Recursive=1")
    Call<MsRepoInfo> getModelFiles(
            @Path("modelGroup") String modelGroup,
            @Path("modelPath") String modelPath
    );
}
