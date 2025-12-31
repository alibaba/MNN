package com.alibaba.mnnllm.android.benchmark

import android.content.Context
import android.os.Build
import android.util.Log
import com.alibaba.mnnllm.android.utils.PreferenceUtils
import com.jaredrummler.android.device.DeviceName
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.logging.HttpLoggingInterceptor
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.TimeUnit
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

/**
 * Service for submitting benchmark results to leaderboard and getting user rankings
 */
class LeaderboardService {
    
    companion object {
        private const val TAG = "LeaderboardService"
        private const val BASE_URL = "https://mnn-mnnchatleaderboard.ms.show/gradio_api/call"
        private const val SUBMIT_ENDPOINT = "$BASE_URL/submit_score"
        private const val RANK_ENDPOINT = "$BASE_URL/get_my_rank"
        
        private const val USER_ID_KEY = "leaderboard_user_id"
        private const val DEFAULT_USER_ID = "anonymous"
    }
    

    private val httpClient = OkHttpClient.Builder()
        .apply {
            if (com.alibaba.mnnllm.android.BuildConfig.DEBUG) {
                 addNetworkInterceptor(com.facebook.stetho.okhttp3.StethoInterceptor())
            }
        }
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .addInterceptor(createLoggingInterceptor())
        .build()
        
    private fun createLoggingInterceptor(): HttpLoggingInterceptor {
        val logging = HttpLoggingInterceptor { message ->
            Log.d(TAG, "🌐 HTTP: $message")
        }
        logging.level = HttpLoggingInterceptor.Level.BODY
        return logging
    }
    
    /**
     * Format JSON string for better readability in logs
     */
    private fun formatJson(jsonString: String): String {
        return try {
            val jsonObject = JSONObject(jsonString)
            jsonObject.toString(2) // Indent with 2 spaces
        } catch (e: Exception) {
            try {
                val jsonArray = JSONArray(jsonString)
                jsonArray.toString(2) // Indent with 2 spaces
            } catch (e2: Exception) {
                // If it's not valid JSON, return as-is with some formatting
                jsonString.replace(",", ",\n   ").replace("{", "{\n   ").replace("}", "\n}")
            }
        }
    }
        
    /**
     * Submit benchmark score to leaderboard
     */
    suspend fun submitScore(
        context: Context,
        modelDisplayName: String,
        prefillSpeed: Double,
        decodeSpeed: Double,
        memoryUsageMb: Double
    ): SubmitResult = withContext(Dispatchers.IO) {
        try {
            // Get device information
            val deviceInfo = getDeviceInfo(context)
            
            // Get user ID
            val userId = getUserId(context)
            
            // Create submission data
            val submissionData = JSONObject().apply {
                put("user_id", userId)
                put("llm_model", modelDisplayName)
                put("device_model", deviceInfo.deviceModel)
                put("device_chipset", deviceInfo.chipset)
                put("device_memory", deviceInfo.memoryMb)
                put("prefill_speed", String.format("%.2f", prefillSpeed))
                put("decode_speed", String.format("%.2f", decodeSpeed))
                put("memory_usage", String.format("%.1f", memoryUsageMb))
            }
            
            val requestData = JSONObject().apply {
                put("data", JSONArray().apply {
                    put(submissionData.toString())
                })
            }
            
            Log.d(TAG, "")
            Log.d(TAG, "🚀 ===== SUBMITTING SCORE TO LEADERBOARD =====")
            Log.d(TAG, "📡 URL: $SUBMIT_ENDPOINT")
            Log.d(TAG, "👤 User ID: $userId")
            Log.d(TAG, "🤖 Model: $modelDisplayName")
            Log.d(TAG, "📊 Prefill Speed: ${String.format("%.2f", prefillSpeed)} tokens/s")
            Log.d(TAG, "⚡ Decode Speed: ${String.format("%.2f", decodeSpeed)} tokens/s")
            Log.d(TAG, "💾 Memory Usage: ${String.format("%.1f", memoryUsageMb)} MB")
            Log.d(TAG, "📱 Device: ${deviceInfo.deviceModel}")
            Log.d(TAG, "🔧 Chipset: ${deviceInfo.chipset}")
            Log.d(TAG, "💿 RAM: ${deviceInfo.memoryMb} MB")
            Log.d(TAG, "")
            Log.d(TAG, "📤 Request JSON:")
            Log.d(TAG, formatJson(requestData.toString()))
            
            val requestBody = requestData.toString().toRequestBody("application/json".toMediaType())
            val request = Request.Builder()
                .url(SUBMIT_ENDPOINT)
                .post(requestBody)
                .addHeader("Content-Type", "application/json")
                .addHeader("User-Agent", "MNN-LLM-Chat Android App")
                .build()
                
            Log.d(TAG, "")
            Log.d(TAG, "📋 Request Headers:")
            request.headers.forEach { header ->
                Log.d(TAG, "   ${header.first}: ${header.second}")
            }
            Log.d(TAG, "📏 Request Body Size: ${requestBody.contentLength()} bytes")
            Log.d(TAG, "")
            Log.d(TAG, "⏳ Sending request...")
            
            val response = httpClient.newCall(request).execute()
            val responseBody = response.body?.string()
            
            Log.d(TAG, "")
            Log.d(TAG, "📥 ===== RESPONSE RECEIVED =====")
            Log.d(TAG, "🏷️  Status: ${response.code} ${response.message}")
            Log.d(TAG, "📋 Response Headers:")
            response.headers.forEach { header ->
                Log.d(TAG, "   ${header.first}: ${header.second}")
            }
            Log.d(TAG, "")
            Log.d(TAG, "📄 Response Body:")
            Log.d(TAG, formatJson(responseBody ?: "null"))
            Log.d(TAG, "")
            
            if (response.isSuccessful) {
                Log.d(TAG, "🎉 ===== SCORE SUBMITTED SUCCESSFULLY =====")
                Log.d(TAG, "✅ Status: Success")
                Log.d(TAG, "🏆 Your benchmark score has been uploaded to the leaderboard!")
                Log.d(TAG, "")
                SubmitResult.Success
            } else {
                val errorMessage = responseBody ?: "Unknown error"
                Log.e(TAG, "")
                Log.e(TAG, "💥 ===== SUBMISSION FAILED =====")
                Log.e(TAG, "❌ Status: HTTP ${response.code} ${response.message}")
                Log.e(TAG, "📝 Error Details:")
                Log.e(TAG, formatJson(errorMessage))
                Log.e(TAG, "")
                Log.e(TAG, "🔍 Common solutions:")
                when (response.code) {
                    400 -> Log.e(TAG, "   • Check request format and data types")
                    401 -> Log.e(TAG, "   • Authentication required or invalid")
                    403 -> Log.e(TAG, "   • Access forbidden - check permissions")
                    404 -> Log.e(TAG, "   • API endpoint not found - check URL")
                    422 -> Log.e(TAG, "   • Invalid data format - check JSON structure")
                    429 -> Log.e(TAG, "   • Rate limit exceeded - try again later")
                    500 -> Log.e(TAG, "   • Server error - try again later")
                    else -> Log.e(TAG, "   • Unexpected error - check network connection")
                }
                Log.e(TAG, "")
                SubmitResult.Error("HTTP ${response.code}: $errorMessage")
            }
        } catch (e: Exception) {
            Log.e(TAG, "")
            Log.e(TAG, "💥 ===== SUBMISSION EXCEPTION =====")
            Log.e(TAG, "❌ Exception Type: ${e.javaClass.simpleName}")
            Log.e(TAG, "📝 Error Message: ${e.message}")
            Log.e(TAG, "🔍 Stack Trace:")
            e.printStackTrace()
            Log.e(TAG, "")
            SubmitResult.Error(e.message ?: "Network error")
        }
    }
    
    /**
     * Get user ranking from leaderboard
     */
    suspend fun getUserRank(
        context: Context,
        modelDisplayName: String
    ): RankResult = withContext(Dispatchers.IO) {
        try {
            val userId = getUserId(context)
            
            // Step 1: Initiate rank query
            val requestData = JSONObject().apply {
                put("data", JSONArray().apply {
                    put(userId)
                    put(modelDisplayName)
                })
            }
            
            Log.d(TAG, "")
            Log.d(TAG, "🏅 ===== GETTING USER RANKING =====")
            Log.d(TAG, "📡 URL: $RANK_ENDPOINT")
            Log.d(TAG, "👤 User ID: $userId")
            Log.d(TAG, "🤖 Model: $modelDisplayName")
            Log.d(TAG, "")
            Log.d(TAG, "📤 Request JSON:")
            Log.d(TAG, formatJson(requestData.toString()))
            
            val requestBody = requestData.toString().toRequestBody("application/json".toMediaType())
            val request = Request.Builder()
                .url(RANK_ENDPOINT)
                .post(requestBody)
                .addHeader("Content-Type", "application/json")
                .addHeader("User-Agent", "MNN-LLM-Chat Android App")
                .build()
                
            Log.d(TAG, "")
            Log.d(TAG, "⏳ Sending ranking request...")
            
            val response = httpClient.newCall(request).execute()
            val responseText = response.body?.string() ?: ""
            
            Log.d(TAG, "")
            Log.d(TAG, "📥 ===== RANKING QUERY RESPONSE =====")
            Log.d(TAG, "🏷️  Status: ${response.code} ${response.message}")
            Log.d(TAG, "📄 Response Body:")
            Log.d(TAG, formatJson(responseText))
            Log.d(TAG, "")
            
            if (!response.isSuccessful) {
                Log.e(TAG, "❌ Rank query failed: ${response.code}")
                Log.e(TAG, "📝 Error: $responseText")
                Log.e(TAG, "")
                return@withContext RankResult.Error("HTTP ${response.code}: $responseText")
            }
            
            // Extract event ID from response
            Log.d(TAG, "🔍 Extracting event ID from response...")
            val eventId = extractEventId(responseText)
            if (eventId == null) {
                Log.e(TAG, "❌ Failed to extract event ID from response")
                Log.e(TAG, "📝 Response doesn't contain valid event_id")
                Log.e(TAG, "")
                return@withContext RankResult.Error("Failed to get event ID")
            }
            
            Log.d(TAG, "✅ Event ID extracted: $eventId")
            Log.d(TAG, "🔄 Starting result polling...")
            Log.d(TAG, "")
            
            // Step 2: Poll for final result
            val rankData = pollRankResult(eventId)
            if (rankData != null) {
                RankResult.Success(rankData)
            } else {
                RankResult.Error("Failed to get rank data")
            }
        } catch (e: Exception) {
            Log.e(TAG, "")
            Log.e(TAG, "💥 ===== RANKING QUERY EXCEPTION =====")
            Log.e(TAG, "❌ Exception Type: ${e.javaClass.simpleName}")
            Log.e(TAG, "📝 Error Message: ${e.message}")
            Log.e(TAG, "🔍 Stack Trace:")
            e.printStackTrace()
            Log.e(TAG, "")
            RankResult.Error(e.message ?: "Network error")
        }
    }
    
    private suspend fun pollRankResult(eventId: String): RankData? = withContext(Dispatchers.IO) {
        try {
            val pollUrl = "$RANK_ENDPOINT/$eventId"
            Log.d(TAG, "")
            Log.d(TAG, "📊 ===== POLLING RANKING RESULT =====")
            Log.d(TAG, "🔗 Poll URL: $pollUrl")
            Log.d(TAG, "🎫 Event ID: $eventId")
            
            val request = Request.Builder()
                .url(pollUrl)
                .get()
                .addHeader("User-Agent", "MNN-LLM-Chat Android App")
                .build()
                
            Log.d(TAG, "⏳ Polling for results...")
            
            val response = httpClient.newCall(request).execute()
            val responseText = response.body?.string() ?: ""
            
            Log.d(TAG, "")
            Log.d(TAG, "📥 ===== POLLING RESPONSE =====")
            Log.d(TAG, "🏷️  Status: ${response.code} ${response.message}")
            Log.d(TAG, "📏 Body Length: ${responseText.length} characters")
            Log.d(TAG, "📄 Response Body:")
            Log.d(TAG, formatJson(responseText))
            Log.d(TAG, "")
            
            if (!response.isSuccessful) {
                Log.e(TAG, "❌ Polling request failed: ${response.code}")
                Log.e(TAG, "📝 Error: $responseText")
                Log.e(TAG, "")
                return@withContext null
            }
            
            // Parse the rank data from response
            Log.d(TAG, "🔍 Parsing ranking data...")
            val rankData = parseRankData(responseText)
            
            if (rankData != null) {
                Log.d(TAG, "")
                Log.d(TAG, "🎯 ===== RANKING RESULTS =====")
                Log.d(TAG, "🏆 Your Rank: ${rankData.rank}")
                Log.d(TAG, "👥 Total Users: ${rankData.totalUsers}")
                Log.d(TAG, "📊 Score: ${rankData.score}")
                Log.d(TAG, "")
            } else {
                Log.w(TAG, "⚠️  Could not parse ranking data from response")
            }
            
            rankData
        } catch (e: Exception) {
            Log.e(TAG, "")
            Log.e(TAG, "💥 ===== POLLING EXCEPTION =====")
            Log.e(TAG, "❌ Exception Type: ${e.javaClass.simpleName}")
            Log.e(TAG, "📝 Error Message: ${e.message}")
            Log.e(TAG, "🔍 Stack Trace:")
            e.printStackTrace()
            Log.e(TAG, "")
            null
        }
    }
    
    private fun extractEventId(responseText: String): String? {
        return try {
            // Look for event_id in the response text
            val regex = """"event_id":"([^"]+)"""".toRegex()
            val matchResult = regex.find(responseText)
            matchResult?.groupValues?.get(1)
        } catch (e: Exception) {
            Log.e(TAG, "Error extracting event ID", e)
            null
        }
    }
    
    private fun parseRankData(responseText: String): RankData? {
        return try {
            val lines = responseText.split('\n')
            var rankData: RankData? = null

            for (line in lines) {
                val trimmed = line.trim()
                if (trimmed.startsWith("data:")) {
                    val jsonPart = trimmed.removePrefix("data:").trim()
                    if (jsonPart.isNotEmpty() && jsonPart != "[null]") {
                        try {
                            val jsonArray = JSONArray(jsonPart)
                            if (jsonArray.length() > 0 && jsonArray.get(0) is JSONObject) {
                                val obj = jsonArray.getJSONObject(0)
                                rankData = RankData(
                                    rank = obj.optInt("rank", 0),
                                    totalUsers = obj.optInt("total_entries_in_model", 0),
                                    score = obj.optDouble("prefill_speed", 0.0)
                                )
                                break
                            }
                        } catch (e: Exception) {
                            try {
                                val jsonObject = JSONObject(jsonPart)
                                if (jsonObject.has("rank")) {
                                    rankData = RankData(
                                        rank = jsonObject.optInt("rank", 0),
                                        totalUsers = jsonObject.optInt("total", 0),
                                        score = jsonObject.optDouble("score", 0.0)
                                    )
                                    break
                                }
                            } catch (_: Exception) {
                            }
                        }
                    }
                }
            }

            if (rankData == null) {
                Log.d(TAG, "No rank data found in response, creating default")
                rankData = RankData(rank = 0, totalUsers = 0, score = 0.0)
            }

            Log.d(TAG, "Parsed rank data: $rankData")
            rankData
        } catch (e: Exception) {
            Log.e(TAG, "Error parsing rank data", e)
            null
        }
    }
    
    private suspend fun getDeviceInfo(context: Context): DeviceInfo = suspendCancellableCoroutine { continuation ->
        Log.d(TAG, "")
        Log.d(TAG, "📱 ===== COLLECTING DEVICE INFO =====")
        DeviceName.with(context).request { info, error ->
            if (error != null) {
                Log.w(TAG, "⚠️  Warning: Failed to get detailed device info", error)
            }
            
            val deviceModel = info?.marketName ?: info?.name ?: Build.MODEL
            val chipset = getChipsetName()
            val memoryMb = getTotalMemoryMb()
            
            Log.d(TAG, "")
            Log.d(TAG, "📱 Device Details:")
            Log.d(TAG, "   🏷️  Model: $deviceModel")
            Log.d(TAG, "   🔧 Chipset: $chipset")
            Log.d(TAG, "   💿 RAM: ${memoryMb} MB")
            Log.d(TAG, "")
            Log.d(TAG, "🔧 Build Information:")
            Log.d(TAG, "   📄 Build.MODEL: ${Build.MODEL}")
            Log.d(TAG, "   🔌 Build.BOARD: ${Build.BOARD}")
            Log.d(TAG, "   🏭 Build.MANUFACTURER: ${Build.MANUFACTURER}")
            Log.d(TAG, "   📱 Build.BRAND: ${Build.BRAND}")
            Log.d(TAG, "   🚀 Build.PRODUCT: ${Build.PRODUCT}")
            Log.d(TAG, "")
            
            val deviceInfo = DeviceInfo(deviceModel, chipset, memoryMb)
            Log.d(TAG, "✅ Device info collected successfully")
            Log.d(TAG, "")
            
            continuation.resume(deviceInfo)
        }
    }
    
    private fun getChipsetName(): String {
        return try {
            // Try to get chipset information from system properties
            when {
                Build.BOARD.contains("sdm", ignoreCase = true) -> "Snapdragon"
                Build.BOARD.contains("exynos", ignoreCase = true) -> "Exynos"
                Build.BOARD.contains("kirin", ignoreCase = true) -> "Kirin"
                Build.BOARD.contains("mt", ignoreCase = true) -> "MediaTek"
                else -> Build.BOARD.ifEmpty { "Unknown" }
            }
        } catch (e: Exception) {
            "Unknown"
        }
    }
    
    private fun getTotalMemoryMb(): Long {
        return try {
            Runtime.getRuntime().maxMemory() / (1024 * 1024)
        } catch (e: Exception) {
            0L
        }
    }
    
    private fun getUserId(context: Context): String {
        return PreferenceUtils.getString(context, USER_ID_KEY, DEFAULT_USER_ID) ?: DEFAULT_USER_ID
    }
    
    /**
     * Set user ID for leaderboard submissions
     */
    fun setUserId(context: Context, userId: String) {
        PreferenceUtils.setString(context, USER_ID_KEY, userId)
    }
    
    /**
     * Test network connectivity to leaderboard server
     */
    suspend fun testNetworkConnectivity(): NetworkTestResult = withContext(Dispatchers.IO) {
        try {
            Log.d(TAG, "")
            Log.d(TAG, "🌐 ===== TESTING NETWORK CONNECTIVITY =====")
            
            val testUrl = BASE_URL.replace("/gradio_api/call", "/")
            Log.d(TAG, "🔗 Target URL: $testUrl")
            Log.d(TAG, "📋 Test Method: HEAD request (lightweight)")
            Log.d(TAG, "⏳ Testing connection...")
            
            val request = Request.Builder()
                .url(testUrl)
                .head() // Use HEAD request to test connectivity without downloading content
                .addHeader("User-Agent", "MNN-LLM-Chat Android App")
                .build()
                
            val response = httpClient.newCall(request).execute()
            
            Log.d(TAG, "")
            Log.d(TAG, "📥 ===== CONNECTIVITY TEST RESULT =====")
            Log.d(TAG, "🏷️  Status: ${response.code} ${response.message}")
            Log.d(TAG, "📋 Response Headers:")
            response.headers.forEach { header ->
                Log.d(TAG, "   ${header.first}: ${header.second}")
            }
            Log.d(TAG, "")
            
            when {
                response.isSuccessful -> {
                    Log.d(TAG, "🎉 ===== CONNECTIVITY TEST PASSED =====")
                    Log.d(TAG, "✅ Network connection to leaderboard server is working!")
                    Log.d(TAG, "🚀 Ready to submit benchmark results")
                    Log.d(TAG, "")
                    NetworkTestResult.Success
                }
                response.code in 400..499 -> {
                    Log.w(TAG, "⚠️  Client error detected: ${response.code}")
                    Log.w(TAG, "📝 This might indicate a request format issue")
                    NetworkTestResult.ClientError(response.code, response.message)
                }
                response.code in 500..599 -> {
                    Log.w(TAG, "⚠️  Server error detected: ${response.code}")
                    Log.w(TAG, "📝 Leaderboard server might be temporarily unavailable")
                    NetworkTestResult.ServerError(response.code, response.message)
                }
                else -> {
                    Log.w(TAG, "⚠️  Unexpected response: ${response.code}")
                    Log.w(TAG, "📝 Unknown server behavior")
                    NetworkTestResult.UnknownError(response.code, response.message)
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "")
            Log.e(TAG, "💥 ===== CONNECTIVITY TEST FAILED =====")
            Log.e(TAG, "❌ Network error: ${e.javaClass.simpleName}")
            Log.e(TAG, "📝 Error message: ${e.message}")
            Log.e(TAG, "🔍 Possible causes:")
            Log.e(TAG, "   • No internet connection")
            Log.e(TAG, "   • DNS resolution failure")
            Log.e(TAG, "   • Firewall blocking the connection")
            Log.e(TAG, "   • Server is completely down")
            Log.e(TAG, "")
            NetworkTestResult.NetworkError(e.message ?: "Unknown network error")
        }
    }
    
    // Data classes
    data class DeviceInfo(
        val deviceModel: String,
        val chipset: String,
        val memoryMb: Long
    )
    
    data class RankData(
        val rank: Int,
        val totalUsers: Int,
        val score: Double
    )
    
    sealed class SubmitResult {
        object Success : SubmitResult()
        data class Error(val message: String) : SubmitResult()
    }
    
    sealed class RankResult {
        data class Success(val rankData: RankData) : RankResult()
        data class Error(val message: String) : RankResult()
    }
    
    sealed class NetworkTestResult {
        object Success : NetworkTestResult()
        data class ClientError(val code: Int, val message: String) : NetworkTestResult()
        data class ServerError(val code: Int, val message: String) : NetworkTestResult()
        data class UnknownError(val code: Int, val message: String) : NetworkTestResult()
        data class NetworkError(val message: String) : NetworkTestResult()
    }
} 