// Created by ruoyi.sjd on 2025/2/13.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.update;


import android.app.AlertDialog;
import android.app.DownloadManager;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Environment;
import android.os.Handler;
import android.os.Looper;
import android.text.TextUtils;
import android.util.Log;
import android.widget.Toast;

import com.alibaba.mnnllm.android.R;
import com.alibaba.mnnllm.android.utils.AppUtils;
import com.alibaba.mnnllm.android.utils.DeviceUtils;
import com.alibaba.mnnllm.android.utils.PreferenceUtils;
import com.alibaba.mnnllm.android.utils.UiUtils;

import org.json.JSONException;
import org.json.JSONObject;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;

import java.io.IOException;
import java.net.URL;

public class UpdateChecker {

    private static final String TAG = "UpdateChecker";
    private final Context context;

    public UpdateChecker(Context context) {
        this.context = context;
    }

    public void checkForUpdates(Context context, boolean forceCheck) {
        if (!forceCheck) {
            long lastCheckTime = PreferenceUtils.getLong(context, "download_last_show_time", 0);
            if (System.currentTimeMillis() - lastCheckTime < 1000 * 60 * 60) {
                return;
            }
        }
        OkHttpClient client = new OkHttpClient();

        Request request = new Request.Builder()
                .url("https://modelscope.cn/datasets/MNN/mnn_llm_app_config/resolve/master/main_config.json")
                .build();

        client.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(Call call, IOException e) {
                if (forceCheck) {
                    UiUtils.showToast(context, context.getString(R.string.get_update_info_failed), Toast.LENGTH_SHORT);
                }
                Log.e(TAG, "get update info failed", e);
            }

            @Override
            public void onResponse(Call call, Response response) throws IOException {
                if (!response.isSuccessful()) {
                    if (forceCheck) {
                        UiUtils.showToast(context, context.getString(R.string.get_update_info_failed), Toast.LENGTH_SHORT);
                    }
                    return;
                }
                String responseData = response.body().string();
                Log.d(TAG, "responde data: " + responseData);
                try {
                    JSONObject jsonObject = new JSONObject(responseData);
                    String latestVersion = jsonObject.getString("latest_version");
                    String updateMessage = jsonObject.getString("update_message");
                    String updateMessageZh = jsonObject.getString("update_message_zh");
                    String downloadUrl = jsonObject.getString("download_url");

                    String currentVersion = AppUtils.INSTANCE.getAppVersionName(context);
                    Log.d(TAG, "currentVersion : " + currentVersion);
                    if (isNewerVersion(latestVersion, currentVersion)) {
                        new Handler(Looper.getMainLooper()).post(() -> {
                            showUpdateDialog(context, latestVersion, updateMessage,updateMessageZh, downloadUrl);
                        });
                    } else if (forceCheck) {
                        UiUtils.showToast(context, context.getString(R.string.no_update), Toast.LENGTH_SHORT);
                    }
                } catch (JSONException e) {
                    Log.e(TAG, "check version error", e);
                }
            }
        });
    }

    private boolean isNewerVersion(String latest, String current) {
        String[] latestParts = latest.split("\\.");
        String[] currentParts = current.split("\\.");
        int length = Math.max(latestParts.length, currentParts.length);
        for (int i = 0; i < length; i++) {
            int latestNum = i < latestParts.length ? Integer.parseInt(latestParts[i]) : 0;
            int currentNum = i < currentParts.length ? Integer.parseInt(currentParts[i]) : 0;
            if (latestNum > currentNum) {
                return true;
            } else if (latestNum < currentNum) {
                return false;
            }
        }
        return false;
    }

    private void showUpdateDialog(Context context, String latestVersion, String updateMessage,
                                  String updateMessageZh, String downloadUrl) {
        PreferenceUtils.setLong(context, "download_last_show_time", System.currentTimeMillis());
        new AlertDialog.Builder(context)
                .setTitle(context.getString(R.string.download_update_available, latestVersion))
                .setMessage(DeviceUtils.isChinese() && !TextUtils.isEmpty(updateMessageZh) ? updateMessageZh : updateMessage)
                .setPositiveButton(R.string.download, (dialog, which) -> {
                    dialog.dismiss();
                    downloadApk(context, downloadUrl);
                })
                .setCancelable(false)
                .setNegativeButton(android.R.string.cancel, (dialog, which) -> dialog.dismiss())
                .show();
    }

    public static String getUrlLastName(String urlStr) {
        try {
            URL url = new URL(urlStr);
            String path = url.getPath();
            String[] pathSegments = path.split("/");
            String lastName = "";
            for (String segment : pathSegments) {
                if (!segment.isEmpty()) {
                    lastName = segment;
                }
            }
            return lastName;
        } catch (Exception e) {
            return null;
        }
    }
    private void downloadApk(Context context, String downloadUrl) {
        String apkName = getUrlLastName(downloadUrl);
        DownloadManager.Request request = new DownloadManager.Request(Uri.parse(downloadUrl))
                .setTitle(apkName)
                .setDescription(context.getString(R.string.wait_install_apk))
                .setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED);
                request.setDestinationInExternalPublicDir(Environment.DIRECTORY_DOWNLOADS, getUrlLastName(downloadUrl));

        DownloadManager downloadManager = (DownloadManager) context.getSystemService(Context.DOWNLOAD_SERVICE);
        downloadManager.enqueue(request);
    }

    private String getVersionName() {
        try {
            String packageName = context.getApplicationContext().getPackageName();
            return context.getApplicationContext().getPackageManager().getPackageInfo(packageName, 0).versionName;
        } catch (PackageManager.NameNotFoundException e) {
            return "99.99";
        }
    }

    public static void installApk(Context context, long downloadId) {
        Log.d(TAG, "installAPK:" + downloadId);
        DownloadManager downloadManager = (DownloadManager) context.getSystemService(Context.DOWNLOAD_SERVICE);
        Uri uri = downloadManager.getUriForDownloadedFile(downloadId);
        Log.d(TAG, "installAPK:" + downloadId + " uri : " + uri);
        if (uri == null) {
            return;
        }
        Intent installIntent = new Intent(Intent.ACTION_INSTALL_PACKAGE);
        installIntent.setDataAndType(uri, "application/vnd.android.package-archive");
        installIntent.setFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION | Intent.FLAG_ACTIVITY_NEW_TASK);
        context.startActivity(installIntent);
    }

}
