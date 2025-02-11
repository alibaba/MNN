// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mls.api;

import java.util.List;
import java.util.concurrent.TimeUnit;

import okhttp3.OkHttpClient;
import okhttp3.logging.HttpLoggingInterceptor;
import retrofit2.*;
import retrofit2.converter.gson.GsonConverterFactory;

public class HfApiClient {

    private static final String TAG = "HfApiClient";

    public static final String HOST_DEFAULT = "huggingface.co";
    public static final String HOST_MIRROR = "hf-mirror.com";

    private String host;
    private HfApiService apiService;
    private OkHttpClient okHttpClient;
    private static HfApiClient sBestClient;

    public static void setBestClient(HfApiClient apiClient) {
        sBestClient = apiClient;
    }

    public static HfApiClient getBestClient() {
        if (sBestClient != null) {
            return sBestClient;
        }
        return null;
    }

    public HfApiClient(String defaultHost) {
        this.host = defaultHost;
        // Initialize Retrofit
        Retrofit retrofit = new Retrofit.Builder()
                .baseUrl("https://" + host)
                .addConverterFactory(GsonConverterFactory.create())
                .client(createOkHttpClient())
                .build();
        apiService = retrofit.create(HfApiService.class);
    }

    public okhttp3.OkHttpClient getOkHttpClient() {
        return okHttpClient;
    }

    private okhttp3.OkHttpClient createOkHttpClient() {
        okhttp3.OkHttpClient.Builder builder = new okhttp3.OkHttpClient.Builder();
        builder.connectTimeout(30, TimeUnit.SECONDS);
        builder.readTimeout(30, TimeUnit.SECONDS);
        okHttpClient =  builder.build();
        return okHttpClient;
    }

    public String getHost() {
        return this.host;
    }

    // Searches repositories based on a keyword
    public void searchRepos(String keyword, RepoSearchCallback callback) {
        Call<List<HfRepoItem>> call = apiService.searchRepos(keyword, "taobao-mnn", 500);
        call.enqueue(new Callback<List<HfRepoItem>>() {
            @Override
            public void onResponse(Call<List<HfRepoItem>> call, Response<List<HfRepoItem>> response) {
                callback.onSuccess(response.body());
            }

            @Override
            public void onFailure(Call<List<HfRepoItem>> call, Throwable t) {
                callback.onFailure(t.getMessage());
            }
        });
    }

    // Retrieves repository information
    public void getRepoInfo(String repoName, String revision, RepoInfoCallback callback) {
        Call<HfRepoInfo> call = apiService.getRepoInfo(repoName, revision);
        call.enqueue(new Callback<HfRepoInfo>() {
            @Override
            public void onResponse(Call<HfRepoInfo> call, Response<HfRepoInfo> response) {
                callback.onSuccess(response.body());
            }

            @Override
            public void onFailure(Call<HfRepoInfo> call, Throwable t) {
                callback.onFailure(t.getMessage());
            }
        });
    }

    // Callback interfaces
    public interface RepoSearchCallback extends CallbackWrapper<List<HfRepoItem>> {
        void onSuccess(List<HfRepoItem> hfRepoItems);
        void onFailure(String error);
    }

    public interface RepoInfoCallback extends CallbackWrapper<HfRepoInfo> {
        void onSuccess(HfRepoInfo hfRepoInfo);
        void onFailure(String error);
    }

    // Wrapper interface to generalize callbacks
    private interface CallbackWrapper<T> {
        void onSuccess(T result);
        void onFailure(String error);
    }

}

