// Created by ruoyi.sjd on 2025/2/10.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mls.api.ms

import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import java.util.concurrent.TimeUnit

class MsApiClient {
    private val host = "modelscope.cn"
    private var okHttpClient: OkHttpClient? = null

    @JvmField
    val apiService: MsApiService

    init {
        // Initialize Retrofit
        val retrofit = Retrofit.Builder()
            .baseUrl("https://$host")
            .addConverterFactory(GsonConverterFactory.create())
            .client(createOkHttpClient())
            .build()
        apiService = retrofit.create(MsApiService::class.java)
    }

    private fun createOkHttpClient(): OkHttpClient? {
        val logging = HttpLoggingInterceptor()
        logging.setLevel(HttpLoggingInterceptor.Level.BODY)
        val builder: OkHttpClient.Builder = OkHttpClient.Builder()
        builder.connectTimeout(30, TimeUnit.SECONDS)
        builder.addInterceptor(logging) //
        builder.readTimeout(30, TimeUnit.SECONDS)
        okHttpClient = builder.build()
        return okHttpClient
    }
}
