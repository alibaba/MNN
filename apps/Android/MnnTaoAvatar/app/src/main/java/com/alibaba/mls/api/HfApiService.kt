// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mls.api

import retrofit2.Call
import retrofit2.http.GET
import retrofit2.http.Path
import retrofit2.http.Query

interface HfApiService {
    // Search repositories
    @GET("/api/models")
    fun searchRepos(
        @Query("search") keyword: String?,
        @Query("author") author: String?,
        @Query("limit") limit: Int,
        @Query("sort") sort: String?
    ): Call<List<ModelItem>>

    @GET("/api/models/{repoName}/revision/{revision}")
    fun getRepoInfo(
        @Path("repoName") repoName: String?,
        @Path("revision") revision: String?
    ): Call<HfRepoInfo?>?
}
