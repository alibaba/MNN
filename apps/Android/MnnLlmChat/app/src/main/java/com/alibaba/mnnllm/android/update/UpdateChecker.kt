// Created by ruoyi.sjd on 2025/2/13.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.update

import android.app.Activity
import android.app.AlertDialog
import android.app.DownloadManager
import android.content.Context
import android.content.DialogInterface
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.os.Handler
import android.os.Looper
import android.provider.Settings
import android.text.TextUtils
import android.util.Log
import android.widget.Toast
import androidx.core.content.FileProvider
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.chat.DownloadReceiver
import com.alibaba.mnnllm.android.utils.AppUtils.getAppVersionName
import com.alibaba.mnnllm.android.utils.DeviceUtils
import com.alibaba.mnnllm.android.utils.PreferenceUtils
import com.alibaba.mnnllm.android.utils.UiUtils
import com.techiness.progressdialoglibrary.ProgressDialog
import okhttp3.Call
import okhttp3.Callback
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Request.Builder
import okhttp3.Response
import org.json.JSONException
import org.json.JSONObject
import java.io.File
import java.io.IOException
import java.net.URL
import kotlin.math.max


class UpdateChecker(private val context: Context) {

    private var manifestUrl = "https://modelscope.cn/datasets/MNN/mnn_llm_app_config/resolve/master/main_config.json"
    private var progressDialog: ProgressDialog? = null

    init {
        if (File("/data/local/tmp/mnn_chat_test").exists()) {
            manifestUrl = manifestUrl.replace("/MNN/", "/JuudeS/")
        }
    }

    fun checkForUpdates(context: Context, forceCheck: Boolean) {
        if (waitingForInstallPermission && (waitingDownloadId > 0 || waitingFileUri != null)) {
            installApk(context, waitingDownloadId, waitingFileUri)
            waitingForInstallPermission = false
            return
        }
        if (!forceCheck) {
            val lastCheckTime = PreferenceUtils.getLong(context, "download_last_show_time", 0)
            if (System.currentTimeMillis() - lastCheckTime < 1000 * 60 * 60) {
                return
            }
        }
        progressDialog = ProgressDialog(context)
        progressDialog?.show()
        val client = OkHttpClient()

        val request: Request = Builder()
            .url(manifestUrl)
            .build()

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                if (forceCheck) {
                    UiUtils.showToast(
                        context,
                        context.getString(R.string.get_update_info_failed),
                        Toast.LENGTH_SHORT
                    )
                }
                Log.e(TAG, "get update info failed", e)
            }

            @Throws(IOException::class)
            override fun onResponse(call: Call, response: Response) {
                if (!response.isSuccessful) {
                    if (forceCheck) {
                        UiUtils.showToast(
                            context,
                            context.getString(R.string.get_update_info_failed),
                            Toast.LENGTH_SHORT
                        )
                    }
                    return
                }
                val responseData = response.body!!.string()
                Log.d(TAG, "responde data: $responseData")
                try {
                    val jsonObject = JSONObject(responseData)
                    val latestVersion = jsonObject.getString("latest_version")
                    val updateMessage = jsonObject.getString("update_message")
                    val updateMessageZh = jsonObject.getString("update_message_zh")
                    val downloadUrl = jsonObject.getString("download_url")

                    val currentVersion = getAppVersionName(context)
                    Log.d(
                        TAG,
                        "currentVersion : $currentVersion"
                    )
                    progressDialog?.dismiss()
                    if (isNewerVersion(latestVersion, currentVersion)) {
                        Handler(Looper.getMainLooper()).post {
                            showUpdateDialog(
                                context,
                                latestVersion,
                                updateMessage,
                                updateMessageZh,
                                downloadUrl
                            )
                        }
                    } else if (forceCheck) {
                        UiUtils.showToast(
                            context,
                            context.getString(R.string.no_update),
                            Toast.LENGTH_SHORT
                        )
                    }
                } catch (e: JSONException) {
                    Log.e(TAG, "check version error", e)
                }
            }
        })
    }

    private fun isNewerVersion(latest: String, current: String): Boolean {
        val latestParts =
            latest.split("\\.".toRegex()).dropLastWhile { it.isEmpty() }.toTypedArray()
        val currentParts =
            current.split("\\.".toRegex()).dropLastWhile { it.isEmpty() }.toTypedArray()
        val length =
            max(latestParts.size.toDouble(), currentParts.size.toDouble()).toInt()
        for (i in 0 until length) {
            val latestNum = if (i < latestParts.size) latestParts[i].toInt() else 0
            val currentNum = if (i < currentParts.size) currentParts[i].toInt() else 0
            if (latestNum > currentNum) {
                return true
            } else if (latestNum < currentNum) {
                return false
            }
        }
        return false
    }

    private fun showUpdateDialog(
        context: Context, latestVersion: String, updateMessage: String,
        updateMessageZh: String, downloadUrl: String
    ) {
        PreferenceUtils.setLong(context, "download_last_show_time", System.currentTimeMillis())
        AlertDialog.Builder(context)
            .setTitle(context.getString(R.string.download_update_available, latestVersion))
            .setMessage(if (DeviceUtils.isChinese && !TextUtils.isEmpty(updateMessageZh)) updateMessageZh else updateMessage)
            .setPositiveButton(R.string.download) { dialog: DialogInterface, which: Int ->
                dialog.dismiss()
                downloadApk(context, downloadUrl)
            }
            .setCancelable(false)
            .setNegativeButton(
                android.R.string.cancel
            ) { dialog: DialogInterface, which: Int -> dialog.dismiss() }
            .show()
    }

    private fun downloadApk(context: Context, downloadUrl: String) {
        val apkName = getUrlLastName(downloadUrl)?: "update.apk"
        val downloadPath = context.getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS)
        val file = File(downloadPath, apkName)
        var downloadFileUri = Uri.fromFile(file)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            val authority = "${context.packageName}.fileprovider"
            downloadFileUri = FileProvider.getUriForFile(context, authority, file)
        }
        if (file.exists()) {
            installApk(context, -9999, downloadFileUri)
            return
        }
        val request = DownloadManager.Request(Uri.parse(downloadUrl))
            .setTitle(apkName)
            .setDescription(context.getString(R.string.wait_install_apk))
            .setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED)
            .setDestinationUri(Uri.fromFile(file))
            .setMimeType("application/vnd.android.package-archive")
        val downloadManager = context.getSystemService(Context.DOWNLOAD_SERVICE) as DownloadManager
        downloadManager.enqueue(request)
    }

    private val versionName: String?
        get() {
            try {
                val packageName = context.applicationContext.packageName
                return context.applicationContext.packageManager
                    .getPackageInfo(packageName, 0).versionName
            } catch (e: PackageManager.NameNotFoundException) {
                return "99.99"
            }
        }

    companion object {
        private var waitingDownloadId: Long = -9999
        private const val TAG = "UpdateChecker"
        var waitingForInstallPermission = false
        private var waitingFileUri: Uri? = null
        fun getUrlLastName(urlStr: String?): String? {
            try {
                val url = URL(urlStr)
                val path = url.path
                val pathSegments =
                    path.split("/".toRegex()).dropLastWhile { it.isEmpty() }.toTypedArray()
                var lastName = ""
                for (segment in pathSegments) {
                    if (!segment.isEmpty()) {
                        lastName = segment
                    }
                }
                return lastName
            } catch (e: Exception) {
                return null
            }
        }

        fun installApk(context: Context, downloadId: Long, fileUri: Uri? = null) {
            if (!context.packageManager.canRequestPackageInstalls()) {
                waitingForInstallPermission = true
                waitingDownloadId = downloadId
                waitingFileUri = fileUri
                Toast.makeText(context, "Please grant permission to install apps.", Toast.LENGTH_LONG).show()
                val intent = Intent(Settings.ACTION_MANAGE_UNKNOWN_APP_SOURCES)
                intent.data = Uri.parse("package:${context.packageName}")
                intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                context.startActivity(intent)
                return
            }
            Log.d(TAG, "installAPK:$fileUri downloadId: $downloadId" )
            val downloadManager =
                context.getSystemService(Context.DOWNLOAD_SERVICE) as DownloadManager
            val uri = fileUri?:downloadManager.getUriForDownloadedFile(downloadId)
            Log.d(TAG, "installAPK:$downloadId uri : $uri")
            if (uri == null) {
                return
            }
            val installIntent = Intent(Intent.ACTION_INSTALL_PACKAGE)
            installIntent.setDataAndType(uri, "application/vnd.android.package-archive")
            installIntent.setFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_ACTIVITY_NEW_TASK)
            context.startActivity(installIntent)
        }

        fun registerDownloadReceiver(context: Context) {
            val downloadCompletedReceiver = DownloadReceiver()
            val filter = IntentFilter(DownloadManager.ACTION_DOWNLOAD_COMPLETE)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                context.registerReceiver(downloadCompletedReceiver, filter, Context.RECEIVER_EXPORTED)
            } else {
                context.registerReceiver(downloadCompletedReceiver, filter)
            }
        }

    }
}
