package com.alibaba.mnnllm.api.openai.manager

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.ClipboardManager
import android.content.ClipData
import android.widget.Toast
import android.net.Uri
import com.alibaba.mnnllm.api.openai.service.OpenAIService
import com.alibaba.mnnllm.api.openai.service.ApiServerConfig
import timber.log.Timber

/**
 * 处理通知栏操作按钮点击事件的广播接收器
 */
class ApiServiceActionReceiver : BroadcastReceiver() {
    
    companion object {
        const val ACTION_STOP_SERVICE = "com.alibaba.mnnllm.api.openai.STOP_SERVICE"
        const val ACTION_COPY_URL = "com.alibaba.mnnllm.api.openai.COPY_URL"
        const val ACTION_TEST_PAGE = "com.alibaba.mnnllm.api.openai.TEST_PAGE"
        const val EXTRA_URL = "extra_url"
    }
    
    override fun onReceive(context: Context, intent: Intent) {
        Timber.tag("ApiServiceActionReceiver").i("=== BROADCAST RECEIVED ===")
        Timber.tag("ApiServiceActionReceiver").i("Action: ${intent.action}")
        Timber.tag("ApiServiceActionReceiver").i("Package: ${intent.`package`}")
        Timber.tag("ApiServiceActionReceiver").i("Component: ${intent.component}")
        Timber.tag("ApiServiceActionReceiver").i("Intent extras: ${intent.extras}")
        Timber.tag("ApiServiceActionReceiver").i("Intent data: ${intent.data}")
        Timber.tag("ApiServiceActionReceiver").i("Intent flags: ${intent.flags}")
        
        when (intent.action) {
            ACTION_STOP_SERVICE -> {
                Timber.tag("ApiServiceActionReceiver").i("Processing STOP_SERVICE action")
                stopApiService(context)
            }
            ACTION_COPY_URL -> {
                Timber.tag("ApiServiceActionReceiver").i("Processing COPY_URL action")
                val url = intent.getStringExtra(EXTRA_URL) ?: "http://localhost:8080"
                Timber.tag("ApiServiceActionReceiver").i("URL to copy: $url")
                copyUrlToClipboard(context, url)
            }
            ACTION_TEST_PAGE -> {
                Timber.tag("ApiServiceActionReceiver").i("Processing TEST_PAGE action")
                val url = intent.getStringExtra(EXTRA_URL) ?: "http://localhost:8080"
                Timber.tag("ApiServiceActionReceiver").i("URL to open: $url")
                openTestPageInBrowser(context, url)
            }
            else -> {
                Timber.tag("ApiServiceActionReceiver").w("Unknown action: ${intent.action}")
                Timber.tag("ApiServiceActionReceiver").w("Expected actions: $ACTION_STOP_SERVICE, $ACTION_COPY_URL, $ACTION_TEST_PAGE")
            }
        }
        
        Timber.tag("ApiServiceActionReceiver").i("=== BROADCAST PROCESSING COMPLETE ===")
    }
    
    /**
     * 停止 API 服务
     */
    private fun stopApiService(context: Context) {
        Timber.tag("ApiServiceActionReceiver").i("Starting to stop API service...")
        try {
            Timber.tag("ApiServiceActionReceiver").i("Calling OpenAIService.releaseService()")
            OpenAIService.releaseService(context)
            Timber.tag("ApiServiceActionReceiver").i("API service stopped successfully")
            Toast.makeText(context, "API Service stopped", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            Timber.tag("ApiServiceActionReceiver").e(e, "Failed to stop API service: ${e.message}")
            Toast.makeText(context, "Failed to stop service", Toast.LENGTH_SHORT).show()
        }
    }
    
    /**
     * 复制 URL 到剪贴板
     */
    private fun copyUrlToClipboard(context: Context, url: String) {
        Timber.tag("ApiServiceActionReceiver").i("Starting to copy URL to clipboard: $url")
        try {
            val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
            Timber.tag("ApiServiceActionReceiver").i("Clipboard service obtained: $clipboard")
            
            val clip = ClipData.newPlainText("API URL", url)
            Timber.tag("ApiServiceActionReceiver").i("ClipData created: $clip")
            
            clipboard.setPrimaryClip(clip)
            Timber.tag("ApiServiceActionReceiver").i("URL copied to clipboard successfully")
            
            val successMessage = context.getString(com.alibaba.mnnllm.android.R.string.api_url_copied)
            Timber.tag("ApiServiceActionReceiver").i("Showing toast: $successMessage")
            Toast.makeText(context, successMessage, Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            Timber.tag("ApiServiceActionReceiver").e(e, "Failed to copy URL to clipboard: ${e.message}")
            Toast.makeText(context, "Failed to copy URL", Toast.LENGTH_SHORT).show()
        }
    }
    
    /**
     * 在浏览器中打开测试页面
     */
    private fun openTestPageInBrowser(context: Context, url: String) {
        Timber.tag("ApiServiceActionReceiver").i("Starting to open test page in browser: $url")
        try {
            // 检查是否启用了认证，如果启用则添加token参数
            val finalUrl = if (ApiServerConfig.isAuthEnabled(context)) {
                val apiKey = ApiServerConfig.getApiKey(context)
                val separator = if (url.contains("?")) "&" else "?"
                val urlWithToken = "$url${separator}token=$apiKey"
                Timber.tag("ApiServiceActionReceiver").i("Auth enabled, adding token to URL: $urlWithToken")
                urlWithToken
            } else {
                Timber.tag("ApiServiceActionReceiver").i("Auth disabled, using original URL")
                url
            }
            
            val intent = Intent(Intent.ACTION_VIEW, Uri.parse(finalUrl))
            intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            Timber.tag("ApiServiceActionReceiver").i("Intent created: $intent")
            
            context.startActivity(intent)
            Timber.tag("ApiServiceActionReceiver").i("Test page opened in browser successfully")
            
            Toast.makeText(context, "Opening test page in browser", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            Timber.tag("ApiServiceActionReceiver").e(e, "Failed to open test page in browser: ${e.message}")
            Toast.makeText(context, "Failed to open browser", Toast.LENGTH_SHORT).show()
        }
    }
}