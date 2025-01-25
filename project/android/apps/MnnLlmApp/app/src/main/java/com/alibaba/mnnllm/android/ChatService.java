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

    private final Map<String, ChatSession> sessionMap = new HashMap<>();

    private static ChatService instance;

    public synchronized ChatSession createSession(String modelDir,
                                                  boolean useTmpPath,
                                                  String sessionId,
                                                  List<ChatDataItem> chatDataItemList) {
        if (TextUtils.isEmpty(sessionId)) {
            sessionId = String.valueOf(System.currentTimeMillis());
        }
        ChatSession session = new ChatSession(sessionId, modelDir, useTmpPath, chatDataItemList);
        sessionMap.put(sessionId, session);
        return session;
    }

    public synchronized ChatSession createDiffusionSession(
                                                  String modelDir,
                                                  String sessionId,
                                                  List<ChatDataItem> chatDataItemList) {
        if (TextUtils.isEmpty(sessionId)) {
            sessionId = String.valueOf(System.currentTimeMillis());
        }
        ChatSession session = new ChatSession(sessionId, modelDir, false, chatDataItemList, true);
        sessionMap.put(sessionId, session);
        return session;
    }

    public synchronized ChatSession getSession(String sessionId) {
        return sessionMap.get(sessionId);
    }

    public synchronized void releaseSession(String sessionId) {
        ChatSession session = sessionMap.remove(sessionId);
        if (session != null) {
            session.release();
        }
    }

    public synchronized void removeSession(String sessionId) {
        sessionMap.remove(sessionId);
    }

    public static synchronized ChatService provide() {
        if (instance == null) {
            instance = new ChatService();
        }
        return instance;
    }

    public synchronized void releaseAllSessions() {
        for (ChatSession session : sessionMap.values()) {
            session.release();
        }
        sessionMap.clear();
    }
}
