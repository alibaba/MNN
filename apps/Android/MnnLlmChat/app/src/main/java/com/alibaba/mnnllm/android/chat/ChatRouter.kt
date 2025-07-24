// Created by ruoyi.sjd on 2025/6/26.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.chat

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.util.Log
import android.widget.Toast
import com.alibaba.mls.api.download.ModelDownloadManager
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.mainsettings.MainSettings.isStopDownloadOnChatEnabled
import com.alibaba.mnnllm.android.model.ModelUtils
import java.io.File

object ChatRouter {

    fun startRun(context: Context, modelIdParam: String, destModelDir:String?, sessionId: String?) {
        Log.d(TAG, "startRun modelIdParam: $modelIdParam destModelDir: $destModelDir sessionId: $sessionId")
        var destPath = destModelDir
        val isDiffusion = ModelUtils.isDiffusionModel(modelIdParam)
        var modelId:String? = modelIdParam
        val downloadManager = ModelDownloadManager.getInstance(context)
        if (!modelIdParam.startsWith("local") && !modelIdParam.contains("/") && !isDiffusion) {
            modelId = ModelUtils.getValidModelIdFromName(downloadManager, modelIdParam)
        }
        Log.d(TAG, "modelId: $modelId")
        if (modelId == null) {
            Toast.makeText(context, context.getString(R.string.model_not_found, modelIdParam), Toast.LENGTH_LONG).show()
            return
        }
        if (isStopDownloadOnChatEnabled(context)) {
            downloadManager.pauseAllDownloads()
        }
        var configFilePath: String?
        if (destPath != null) {
            configFilePath = destPath
        } else {
            configFilePath = ModelUtils.getConfigPathForModel(modelId)
        }
        val configFileExists = configFilePath?.let { File(it).exists() } ?: false
        if (!configFileExists) {
            Toast.makeText(
                context,
                context.getString(R.string.config_file_not_found, configFilePath?: modelId),
                Toast.LENGTH_LONG
            ).show()
            return
        }
        Log.d(TAG, "isDiffusion: ${isDiffusion}, configFilePath: $configFilePath")
        val intent = Intent(context, ChatActivity::class.java)
        intent.putExtra("chatSessionId", sessionId)
        if (isDiffusion) {
            intent.putExtra("diffusionDir", configFilePath)
        } else {
            intent.putExtra("configFilePath", configFilePath)
        }
        intent.putExtra("modelId", modelId)
        intent.putExtra("modelName", ModelUtils.getModelName(modelId))

        if (context !is Activity) {
            intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
        }
        context.startActivity(intent)
    }

    private const val TAG = "ChatRouter"

}