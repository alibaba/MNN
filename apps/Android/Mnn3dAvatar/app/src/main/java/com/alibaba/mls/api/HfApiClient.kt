// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mls.api

import okhttp3.OkHttpClient
import retrofit2.Call
import retrofit2.Callback
import retrofit2.Response
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import java.util.concurrent.TimeUnit

class HfApiClient(@JvmField val host: String) {
    private val apiService: HfApiService
    var okHttpClient: OkHttpClient? = null
        private set

    init {
        // Initialize Retrofit
        val retrofit = Retrofit.Builder()
            .baseUrl("https://$host")
            .addConverterFactory(GsonConverterFactory.create())
            .client(createOkHttpClient()!!)
            .build()
        apiService = retrofit.create(HfApiService::class.java)
    }

    private fun createOkHttpClient(): OkHttpClient? {
        val builder: OkHttpClient.Builder = OkHttpClient.Builder()
        builder.connectTimeout(30, TimeUnit.SECONDS)
        builder.readTimeout(30, TimeUnit.SECONDS)
//        val okHttpClient:OkHttpClient = builder.build()
        okHttpClient = builder.build()
        return okHttpClient
    }

    // Searches repositories based on a keyword
    fun searchRepos(keyword: String?, callback: RepoSearchCallback) {
        val call = apiService.searchRepos(keyword, "taobao-mnn", 500, "downloads")
        call.enqueue(object : Callback<List<ModelItem>> {
            override fun onResponse(
                call: Call<List<ModelItem>>,
                response: Response<List<ModelItem>>
            ) {
                callback.onSuccess(response.body()!!)
            }

            override fun onFailure(call: Call<List<ModelItem>>, t: Throwable) {
                callback.onFailure(t.message)
            }
        })
    }

    // Retrieves repository information
    fun getRepoInfo(repoName: String?, revision: String?, callback: RepoInfoCallback) {
        val call = apiService.getRepoInfo(repoName, revision)
        call!!.enqueue(object : Callback<HfRepoInfo?> {
            override fun onResponse(call: Call<HfRepoInfo?>, response: Response<HfRepoInfo?>) {
                callback.onSuccess(response.body())
            }

            override fun onFailure(call: Call<HfRepoInfo?>, t: Throwable) {
                callback.onFailure(t.message)
            }
        })
    }

    // Callback interfaces
    interface RepoSearchCallback :
        CallbackWrapper<List<ModelItem>> {
        override fun onSuccess(hfModelItems: List<ModelItem>)
        override fun onFailure(error: String?)
    }

    interface RepoInfoCallback : CallbackWrapper<HfRepoInfo?> {
        override fun onSuccess(hfRepoInfo: HfRepoInfo?)
        override fun onFailure(error: String?)
    }

    // Wrapper interface to generalize callbacks
    interface CallbackWrapper<T> {
        fun onSuccess(result: T)
        fun onFailure(error: String?)
    }

    companion object {
        private const val TAG = "HfApiClient"

        const val HOST_DEFAULT: String = "huggingface.co"
        const val HOST_MIRROR: String = "hf-mirror.com"

        private var sBestClient: HfApiClient? = null

        @JvmStatic
        var bestClient: HfApiClient?
            get() {
                if (sBestClient != null) {
                    return sBestClient
                }
                return null
            }
            set(apiClient) {
                sBestClient = apiClient
            }
    }
}

