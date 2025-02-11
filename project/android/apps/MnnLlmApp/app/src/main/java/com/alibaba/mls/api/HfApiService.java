// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mls.api;

import java.util.List;

import retrofit2.Call;
import retrofit2.http.GET;
import retrofit2.http.Path;
import retrofit2.http.Query;

public interface HfApiService {

    // Search repositories
    @GET("/api/models")
    Call<List<HfRepoItem>> searchRepos(
            @Query("search") String keyword,
            @Query("author") String author,
            @Query("limit") int limit
    );

    // Get repository information
    @GET("/api/models/{repoName}/revision/{revision}")
    Call<HfRepoInfo> getRepoInfo(
            @Path("repoName") String repoName,
            @Path("revision") String revision
    );
}
