// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mls.api

import okhttp3.OkHttpClient
import retrofit2.Call
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import java.util.concurrent.TimeUnit

/**
 * Minimal HuggingFace API client for the framework
 * Only includes repo info fetching, no model search functionality
 */
class HfApiClient(val host: String) {
    private val apiService: HfApiService
    var okHttpClient: OkHttpClient? = null
        private set

    init {
        // Initialize Retrofit
        val retrofit = Retrofit.Builder()
            .baseUrl("https://$host")
            .addConverterFactory(GsonConverterFactory.create())
            .client(createOkHttpClient())
            .build()
        apiService = retrofit.create(HfApiService::class.java)
    }

    private fun createOkHttpClient(): OkHttpClient? {
        val builder: OkHttpClient.Builder = OkHttpClient.Builder()
        builder.connectTimeout(30, TimeUnit.SECONDS)
        builder.readTimeout(30, TimeUnit.SECONDS)
        okHttpClient = builder.build()
        return okHttpClient
    }

    // Get repo file tree
    fun getRepoTree(repoId: String, revision: String = "main"): Call<HfRepoInfo> {
        return apiService.getRepoTree(repoId, revision)
    }

    companion object {
        const val HOST_DEFAULT = "huggingface.co"
        const val HOST_CN = "hf-mirror.com"

        val bestClient: HfApiClient?
            get() = HfApiClient(HOST_CN)  // Default to CN mirror for better connectivity
    }
}
