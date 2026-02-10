// Created by ruoyi.sjd on 2025/5/13.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mls.api.ml

import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Call
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.GET
import retrofit2.http.Path
import retrofit2.http.Query
import java.util.concurrent.TimeUnit

class MlApiClient {
    private val host = "modelers.cn"
    private var okHttpClient: OkHttpClient? = null

    val apiService: MlApiService

    init {
        val retrofit = Retrofit.Builder()
            .baseUrl("https://$host/")
            .addConverterFactory(GsonConverterFactory.create())
            .client(createOkHttpClient())
            .build()
        apiService = retrofit.create(MlApiService::class.java)
    }

    private fun createOkHttpClient(): OkHttpClient? {
        val logging = HttpLoggingInterceptor()
        logging.setLevel(HttpLoggingInterceptor.Level.BODY)
        val builder: OkHttpClient.Builder = OkHttpClient.Builder()
        builder.connectTimeout(30, TimeUnit.SECONDS)
        builder.addInterceptor(logging)
        builder.readTimeout(30, TimeUnit.SECONDS)
        okHttpClient = builder.build()
        return okHttpClient
    }

    interface MlApiService {
        @GET("api/v1/file/{modelGroup}/{modelPath}")
        fun getModelFiles(
            @Path("modelGroup") modelGroup: String,
            @Path("modelPath") modelPath: String,
            @Query("path") path: String,
        ): Call<MlRepoInfo>
    }
}
