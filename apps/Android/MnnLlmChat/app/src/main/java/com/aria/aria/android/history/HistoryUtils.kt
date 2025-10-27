// Created by ruoyi.sjd on 2025/1/14.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.history

import android.content.Context
import android.util.Log
import com.alibaba.mls.api.download.DownloadFileUtils
import com.alibaba.mnnllm.android.chat.model.ChatDataManager
import com.alibaba.mnnllm.android.utils.FileUtils.getSessionResourceBasePath
import java.io.File

object HistoryUtils {
    const val TAG: String = "ChatHistoryFragment"

    fun deleteHistory(
        context: Context,
        chatDataManager: ChatDataManager,
        historySessionId: String
    ) {
        Log.d(TAG, "delete historySessionId: $historySessionId")
        chatDataManager.deleteSession(historySessionId)
        val sessionResourceDir = File(getSessionResourceBasePath(context, historySessionId))
        DownloadFileUtils.deleteDirectoryRecursively(sessionResourceDir)
    }
}
