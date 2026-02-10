// Created by ruoyi.sjd on 2025/4/22.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.benchmark

import android.content.Context
import android.util.Log
import android.widget.Toast
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.MainScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.File
import java.io.IOException
import java.util.concurrent.TimeUnit

class BenchmarkTest(private val context:Context) {

    object config {
        val rootPath = "/data/local/tmp/mnn_bench"
        val testResultJsonl1 = "$rootPath/test_results1.jsonl"
        val testResultJsonl2 = "$rootPath/test_results2.jsonl"
    }

    var uploadInfoJsonString:String? = null

    data class UploadConfig(
        val uploadUrl: String,
        val sessionId: String
    )

    fun parseUploadInfo(jsonString: String?): UploadConfig? {
        if (jsonString == null) return null
        return try {
            val jsonObject = JSONObject(jsonString)
            UploadConfig(
                uploadUrl = jsonObject.getString("upload_url"),
                sessionId = jsonObject.getString("session_id")
            )
        } catch (e: Exception) {
            Log.e(TAG, "Failed to parse uploadInfo JSON", e)
            null
        }
    }

    private val client by lazy {
        OkHttpClient.Builder()
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(30, TimeUnit.SECONDS)
            .writeTimeout(30, TimeUnit.SECONDS)
            .retryOnConnectionFailure(true)
            .build()
    }

    fun uploadJsonl() {
        uploadJsonlFiles(config.testResultJsonl1, config.testResultJsonl2)
    }

    fun uploadJsonlFiles(file1Path: String, file2Path: String) {
//        uploadInfoJsonString = """{"upload_url":"http://192.168.1.105:5001/api/mobile-upload","session_id":"53c12c16-d347-4298-85ef-d2e5a73c1326"}"""
        Log.d(TAG, "file $uploadInfoJsonString")
        val currentUploadConfig = parseUploadInfo(uploadInfoJsonString)
        if (currentUploadConfig == null) {
            Log.e(TAG, "Upload configuration is invalid or not set.")
            // Optionally show a toast message here if on UI thread or switch context
            // toast("Upload configuration error.")
            return
        }

        CoroutineScope(Dispatchers.IO).launch {
            try {
                val file1 = File(file1Path)
                val file2 = File(file2Path)

                if (!file1.exists()) {
                    Log.e(TAG, "File 1 not found at: $file1Path")
                    withContext(Dispatchers.Main) { toast("Error: File 1 not found.") }
                    return@launch
                }
                if (!file2.exists()) {
                    Log.e(TAG, "File 2 not found at: $file2Path")
                    withContext(Dispatchers.Main) { toast("Error: File 2 not found.") }
                    return@launch
                }

                Log.d(TAG, "Starting upload to: ${currentUploadConfig.uploadUrl} with session ID: ${currentUploadConfig.sessionId}")
                val file1RequestBody = file1.asRequestBody("application/octet-stream".toMediaTypeOrNull())
                val multipartBody = MultipartBody.Builder()
                    .setType(MultipartBody.FORM)
                    .addFormDataPart("session_id", currentUploadConfig.sessionId)
                    .addFormDataPart("uploaded_file", file1.name, file1RequestBody) // "file1" is the name backend expects
                    .build()
                val request = Request.Builder()
                    .url(currentUploadConfig.uploadUrl)
                    .post(multipartBody)
                    .build()
                val response = client.newCall(request).execute()
                if (response.isSuccessful) {
                    val responseBodyString = response.body?.string()
                    Log.d(TAG, "Upload successful! Server response: $responseBodyString")
                    withContext(Dispatchers.Main) {
                        toast("Upload successful!")
                    }
                } else {
                    val errorBodyString = response.body?.string()
                    Log.e(TAG, "Upload failed. Server error ${response.code}: $errorBodyString")
                    withContext(Dispatchers.Main) {
                        toast("Upload failed: ${response.code} - ${errorBodyString ?: "Unknown server error"}")
                    }
                }
                response.close()
            } catch (e: IOException) {
                Log.e(TAG, "Upload failed due to IOException:", e)
                withContext(Dispatchers.Main) {
                    toast("Upload failed: Network error - ${e.message}")
                }
            } catch (e: IllegalStateException) { // Catch errors like "URL not set"
                Log.e(TAG, "Upload failed due to IllegalStateException:", e)
                withContext(Dispatchers.Main) {
                    toast("Upload error: ${e.message}")
                }
            } catch (e: Exception) { // Catch any other exceptions
                Log.e(TAG, "An unexpected error occurred during upload:", e)
                withContext(Dispatchers.Main) {
                    toast("Upload error: An unexpected error occurred - ${e.message}")
                }
            }
        }
    }

    fun uploadJsonlx() {
//        192.168.1.105:5001
        uploadInfoJsonString = """{"upload_url":"http://192.168.1.105:5001/api/mobile-upload","session_id":"53c12c16-d347-4298-85ef-d2e5a73c1326"}"""
        CoroutineScope(Dispatchers.IO).launch {
            try {
                val url = uploadInfoJsonString ?: throw IllegalStateException("上传URL未设置")
                Log.d(TAG, "upload begin uploadUrl : $uploadInfoJsonString")
                val fileContent = File(config.testResultJsonl2).readText(Charsets.UTF_8)
                val body = fileContent.toRequestBody("application/json; charset=utf-8".toMediaType())
                val request = Request.Builder()
                    .url(url)
                    .post(body)
                    .build()
                try {
                    val response = client.newCall(request).execute()
                    if (response.isSuccessful) {
                        Log.d(TAG, "上传成功！服务器返回: ${response.body?.string()}")
                    } else {
                        Log.e(TAG, "服务器错误 ${response.code}: ${response.body?.string()}")
                    }
                } catch (e: IOException) {
                    withContext(Dispatchers.Main) {
                        Log.e(TAG, "上传失败:", e)
                        toast("上传失败: ${e.message}")
                    }
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    Log.e(TAG, "上传过程中发生错误:", e)
                    toast("上传错误: ${e.message}")
                }
            }
        }
    }

    private fun toast(msg: String) {
        MainScope().launch {
            Toast.makeText(context, msg, Toast.LENGTH_SHORT).show()
        }
    }

    companion object {
        private const val TAG = "BenchmarkTest"
    }
}