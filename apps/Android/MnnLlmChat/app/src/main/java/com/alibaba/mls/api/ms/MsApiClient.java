// Created by ruoyi.sjd on 2025/2/10.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mls.api.ms;

import java.util.concurrent.TimeUnit;

import okhttp3.OkHttpClient;
import okhttp3.logging.HttpLoggingInterceptor;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

public class MsApiClient {
    private String host = "modelscope.cn";
    private OkHttpClient okHttpClient;

    private MsApiService apiService;

    public MsApiClient() {
        // Initialize Retrofit
        Retrofit retrofit = new Retrofit.Builder()
                .baseUrl("https://" + host)
                .addConverterFactory(GsonConverterFactory.create())
                .client(createOkHttpClient())
                .build();
        apiService = retrofit.create(MsApiService.class);
    }

    public MsApiService getApiService() {
        return apiService;
    }

    private okhttp3.OkHttpClient createOkHttpClient() {
        HttpLoggingInterceptor logging = new HttpLoggingInterceptor();
        logging.setLevel(HttpLoggingInterceptor.Level.BODY);
        okhttp3.OkHttpClient.Builder builder = new okhttp3.OkHttpClient.Builder();
        builder.connectTimeout(30, TimeUnit.SECONDS);
        builder.addInterceptor(logging); //
        builder.readTimeout(30, TimeUnit.SECONDS);
        okHttpClient =  builder.build();
        return okHttpClient;
    }
}
