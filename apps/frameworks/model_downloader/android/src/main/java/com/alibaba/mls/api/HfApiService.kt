// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mls.api

import retrofit2.Call
import retrofit2.http.GET
import retrofit2.http.Path
import retrofit2.http.Query

/**
 * Minimal HuggingFace API service for the framework
 */
interface HfApiService {
    @GET("/api/models/{repoId}/tree/{revision}")
    fun getRepoTree(
        @Path("repoId", encoded = true) repoId: String,
        @Path("revision") revision: String,
        @Query("recursive") recursive: Boolean = true,
        @Query("expand") expand: Boolean = true
    ): Call<HfRepoInfo>
}
