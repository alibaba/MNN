// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android;

import android.util.Log;

import com.alibaba.mnnllm.android.chat.ChatDataItem;
import com.alibaba.mnnllm.android.utils.ModelUtils;

import java.io.Serializable;
import java.util.HashMap;
import java.util.List;
import java.util.stream.Collectors;

public class ChatSession implements Serializable {

    private final String configPath;
    private long nativePtr;

    public static final String TAG = "ChatSession";

    private String sessionId;

    private volatile boolean mGenerating = false;
    private volatile boolean mReleaseRequeted = false;

    private List<ChatDataItem> savedHistory;
    private boolean isDiffusion;

    private boolean useTmpPath;
    private boolean keepHistory;

    public ChatSession(String sessionId, String configPath, boolean useTmpPath, List<ChatDataItem> history) {
        this(sessionId, configPath, useTmpPath, history, false);
    }

    public ChatSession(String sessionId, String configPath, boolean useTmpPath, List<ChatDataItem> history, boolean isDiffusion) {
        this.sessionId = sessionId;
        this.configPath = configPath;
        this.savedHistory = history;
        this.isDiffusion = isDiffusion;
        this.useTmpPath = useTmpPath;
    }

    public void load() {
        Log.d(TAG, "MNN_DEBUG load begin");
        List<String> historyStringList = null;
        if (this.savedHistory != null && !this.savedHistory.isEmpty()) {
            historyStringList = this.savedHistory.stream().map(ChatDataItem::getText).collect(Collectors.toList());
        }
        nativePtr = initNative(configPath, useTmpPath, historyStringList, isDiffusion);
    }

    public List<ChatDataItem> getSavedHistory() {
        return savedHistory;
    }

    public String generateNewSession() {
        this.sessionId = String.valueOf(System.currentTimeMillis());
        return this.sessionId;
    }

    public String getSessionId() {
        return sessionId;
    }

    public HashMap<String, Object> generate(String input, GenerateProgressListener progressListener) {
        synchronized (this) {
            Log.d(TAG, "MNN_DEBUG submit" + input);
            mGenerating = true;
            HashMap<String, Object> result = submitNative(nativePtr, input, keepHistory, progressListener);
            mGenerating = false;
            if (mReleaseRequeted) {
                releaseInner();
            }
            return result;
        }
    }

    public HashMap<String, Object> generateDiffusion(String input, String output, GenerateProgressListener progressListener) {
        synchronized (this) {
            Log.d(TAG, "MNN_DEBUG submit" + input);
            mGenerating = true;
            HashMap<String, Object> result = submitDiffusionNative(nativePtr, input, output, progressListener);
            mGenerating = false;
            if (mReleaseRequeted) {
                releaseInner();
            }
            return result;
        }
    }

    public void reset() {
        synchronized (this) {
            resetNative(nativePtr);
        }
    }

    public void release() {
        synchronized (this) {
            Log.d(TAG, "MNN_DEBUG release" + nativePtr);
            if (!mGenerating) {
                releaseInner();
            } else {
                mReleaseRequeted = true;
            }
        }
    }

    private void releaseInner() {
        if (nativePtr > 0) {
            releaseNative(nativePtr, isDiffusion);
            nativePtr = 0;
            ChatService.provide().removeSession(sessionId);
        }
    }

    @Override
    protected void finalize() throws Throwable {
        super.finalize();
        release();
    }

    public native long initNative(String configPath, boolean useTmpPath, List<String> history, boolean isDiffusion);
    private native HashMap<String, Object> submitNative(long instanceId, String input, boolean keepHistory, GenerateProgressListener listener);

    private native HashMap<String, Object> submitDiffusionNative(long instanceId, String input, String outputPath, GenerateProgressListener progressListener);
    private native void resetNative(long instanceId);

    private native void releaseNative(long instanceId, boolean isDiffusion);

    static {
        System.loadLibrary("llm");
        System.loadLibrary("MNN_CL");
    }

    public void setKeepHistory(boolean keepHistory) {
        this.keepHistory = keepHistory;
    }

    public interface GenerateProgressListener {
        boolean onProgress(String progress);
    }
}
