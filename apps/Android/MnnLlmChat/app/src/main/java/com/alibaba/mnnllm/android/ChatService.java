// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android;

import android.text.TextUtils;

import com.alibaba.mnnllm.android.chat.ChatDataItem;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ChatService{

    static {
        System.loadLibrary("mnnllmapp");
    }

    private final Map<String, ChatSession> transformerSessionMap = new HashMap<>();
    private final Map<String, ChatSession> diffusionSessionMap = new HashMap<>();

    private static ChatService instance;

    public synchronized ChatSession createSession(String modelId,
                                                  String modelDir,
                                                  boolean useTmpPath,
                                                  String sessionId,
                                                  List<ChatDataItem> chatDataItemList) {
        if (TextUtils.isEmpty(sessionId)) {
            sessionId = String.valueOf(System.currentTimeMillis());
        }
        ChatSession session = new ChatSession(modelId, sessionId, modelDir, useTmpPath, chatDataItemList);
        transformerSessionMap.put(sessionId, session);
        return session;
    }

    public synchronized ChatSession createDiffusionSession(
                                                  String modelId,
                                                  String modelDir,
                                                  String sessionId,
                                                  List<ChatDataItem> chatDataItemList) {
        if (TextUtils.isEmpty(sessionId)) {
            sessionId = String.valueOf(System.currentTimeMillis());
        }
        ChatSession session = new ChatSession(modelId, sessionId, modelDir, false, chatDataItemList, true);
        diffusionSessionMap.put(sessionId, session);
        return session;
    }

    public synchronized ChatSession getSession(String sessionId) {
        if (transformerSessionMap.containsKey(sessionId)) {
            return transformerSessionMap.get(sessionId);
        } else {
            return diffusionSessionMap.get(sessionId);
        }
    }

    public synchronized void removeSession(String sessionId) {
        transformerSessionMap.remove(sessionId);
        diffusionSessionMap.remove(sessionId);
    }

    public static synchronized ChatService provide() {
        if (instance == null) {
            instance = new ChatService();
        }
        return instance;
    }
}
