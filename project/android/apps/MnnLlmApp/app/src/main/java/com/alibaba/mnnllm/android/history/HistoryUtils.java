// Created by ruoyi.sjd on 2025/1/14.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.history;

import android.content.Context;
import android.util.Log;

import com.alibaba.mls.api.download.HfFileUtils;
import com.alibaba.mnnllm.android.chat.ChatDataManager;
import com.alibaba.mnnllm.android.utils.FileUtils;

import java.io.File;

public class HistoryUtils {
    public static final String TAG = "ChatHistoryFragment";

    public static void deleteHistory(Context context, ChatDataManager chatDataManager, String historySessionId) {
        Log.d(TAG, "delete historySessionId: " + historySessionId);
        chatDataManager.deleteSession(historySessionId);
        File sessionResourceDir = new File(FileUtils.getSessionResourceBasePath(context, historySessionId));
        HfFileUtils.deleteDirectoryRecursively2(sessionResourceDir);
    }
}
