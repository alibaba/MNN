// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android;

import android.util.Log;

import com.alibaba.mls.api.ApplicationProvider;
import com.alibaba.mnnllm.android.chat.ChatDataItem;
import com.alibaba.mnnllm.android.mainsettings.MainSettings;
import com.alibaba.mnnllm.android.utils.FileUtils;
import com.alibaba.mnnllm.android.utils.ModelPreferences;
import com.alibaba.mnnllm.android.utils.ModelUtils;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.util.HashMap;
import java.util.List;
import java.util.stream.Collectors;

public class ChatSession {

    private final String configPath;
    private long nativePtr;

    public static final String TAG = "ChatSession";

    private String sessionId;

    private volatile boolean modelLoading = false;

    private volatile boolean mGenerating = false;
    private volatile boolean mReleaseRequeted = false;

    private List<ChatDataItem> savedHistory;
    private boolean isDiffusion;

    private boolean useTmpPath;
    private boolean keepHistory;
    private String modelId;

    public ChatSession(String modelId, String sessionId, String configPath, boolean useTmpPath, List<ChatDataItem> history) {
        this(modelId, sessionId, configPath, useTmpPath, history, false);
    }

    public ChatSession(String modelId, String sessionId, String configPath, boolean useTmpPath, List<ChatDataItem> history, boolean isDiffusion) {
        this.modelId = modelId;
        this.sessionId = sessionId;
        this.configPath = configPath;
        this.savedHistory = history;
        this.isDiffusion = isDiffusion;
        this.useTmpPath = useTmpPath;
    }

    public void load() {
        Log.d(TAG, "MNN_DEBUG load begin");
        modelLoading = true;
        List<String> historyStringList = null;
        if (this.savedHistory != null && !this.savedHistory.isEmpty()) {
            historyStringList = this.savedHistory.stream().map(ChatDataItem::getText).collect(Collectors.toList());
        }
        String rootCacheDir = "";
        if (ModelPreferences.useMmap(ApplicationProvider.get(), modelId)) {
            rootCacheDir = FileUtils.getMmapDir(modelId, configPath.contains("modelscope"));
            new File(rootCacheDir).mkdirs();
        }
        boolean useOpencl = ModelPreferences.getBoolean(ApplicationProvider.get(), modelId, ModelPreferences.KEY_BACKEND, false);
        String backend = useOpencl ? "opencl" : "cpu";
        String sampler = ModelPreferences.getString(ApplicationProvider.get(), modelId, ModelPreferences.KEY_SAMPLER, "greedy");
        JSONObject configJson = new JSONObject();
        try {
            configJson.put("backend", backend);
            configJson.put("sampler", sampler);
            configJson.put("is_diffusion", isDiffusion);
            configJson.put("is_r1", ModelUtils.isR1Model(modelId));
            configJson.put("diffusion_memory_mode", MainSettings.INSTANCE.getDiffusionMemoryMode(ApplicationProvider.get()));
        } catch (JSONException e) {
            throw new RuntimeException(e);
        }
        nativePtr = initNative(rootCacheDir, modelId, configPath,
                useTmpPath, historyStringList, configJson.toString());
        modelLoading = false;
        if (mReleaseRequeted) {
            release();
        }
    }

    public String getDebugInfo() {
        return getDebugInfoNative(nativePtr) + "\n";
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
                release();
            }
            return result;
        }
    }

    public HashMap<String, Object> generateDiffusion(String input, String output, int iterNum, int randomSeed, GenerateProgressListener progressListener) {
        synchronized (this) {
            Log.d(TAG, "MNN_DEBUG submit" + input);
            mGenerating = true;
            HashMap<String, Object> result = submitDiffusionNative(nativePtr, input, output, iterNum, randomSeed, progressListener);
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
            Log.d(TAG, "MNN_DEBUG release nativePtr: " + nativePtr + " mGenerating: " + mGenerating);
            if (!mGenerating && !modelLoading) {
                releaseInner();
            } else {
                mReleaseRequeted = true;
                while (mGenerating || modelLoading) {
                    try {
                        wait();
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        Log.e(TAG, "Thread interrupted while waiting for release", e);
                    }
                }
                releaseInner();
            }
        }
    }
    private void releaseInner() {
        if (nativePtr != 0) {
            releaseNative(nativePtr, isDiffusion);
            nativePtr = 0;
            ChatService.provide().removeSession(sessionId);
            notifyAll();
        }
    }

    @Override
    protected void finalize() throws Throwable {
        super.finalize();
        release();
    }

    public native long initNative(String rootCacheDir,
                                  String modelId,
                                  String configPath,
                                  boolean useTmpPath,
                                  List<String> history,
                                  String configJsonStr);
    private native HashMap<String, Object> submitNative(long instanceId, String input, boolean keepHistory, GenerateProgressListener listener);

    private native HashMap<String, Object> submitDiffusionNative(long instanceId, String input, String outputPath, int iterNum, int randomSeed, GenerateProgressListener progressListener);
    private native void resetNative(long instanceId);

    private native String getDebugInfoNative(long instanceId);

    private native void releaseNative(long instanceId, boolean isDiffusion);

    static {
        System.loadLibrary("llm");
        System.loadLibrary("MNN_CL");
    }

    public void setKeepHistory(boolean keepHistory) {
        this.keepHistory = keepHistory;
    }

    public void clearMmapCache() {
        FileUtils.clearMmapCache(modelId);
    }

    public interface GenerateProgressListener {
        boolean onProgress(String progress);
    }
}
