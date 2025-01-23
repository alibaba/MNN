// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.chat;

import android.net.Uri;

import java.io.File;

public class ChatDataItem {
    private String time;
    public AudioPlayerComponent audioPlayComponent;
    private String text;
    private int type;
    private Uri imageUri;

    private Uri audioUri;

    private String benchmarkInfo;

    private float audioDuration;

    public ChatDataItem(String time, int type, String text) {
        this.time = time;
        this.type = type;
        this.text = text;
    }

    public ChatDataItem(int type) {
        this.type = type;
    }

    public String getTime() {
        return time;
    }

    public int getType() {
        return type;
    }

    public String getText() {
        return text;
    }

    public void setText(String text) {
        this.text = text;
    }

    public Uri getImageUri() {
        return imageUri;
    }

    public void setImageUri(Uri image) {
        imageUri = image;
    }

    public void setTime(String time) {
        this.time = time;
    }

    public String getBenchmarkInfo() {
        return benchmarkInfo;
    }

    public void setBenchmarkInfo(String benchmarkInfo) {
        this.benchmarkInfo = benchmarkInfo;
    }

    public static ChatDataItem createImageInputData(String timeString, String text, Uri imageUri) {
        ChatDataItem result = new ChatDataItem(timeString, ChatViewHolders.USER, text);
        result.setImageUri(imageUri);
        return result;
    }

    public static ChatDataItem createAudioInputData(String timeString, String text, String audioPath, float duration) {
        ChatDataItem result = new ChatDataItem(timeString, ChatViewHolders.USER, text);
        result.setAudioUri(Uri.fromFile(new File(audioPath)));
        result.audioDuration = duration;
        return result;
    }

    public Uri getAudioUri() {
        return audioUri;
    }

    public void setAudioUri(Uri audioUri) {
        this.audioUri = audioUri;
    }

    public float getAudioDuration() {
        return audioDuration;
    }

    public void setAudioDuration(float audioDuration) {
        this.audioDuration = audioDuration;
    }

    public String getAudioPath() {
        if (this.audioUri != null && "file".equals(this.audioUri.getScheme())) {
            return this.audioUri.getPath();
        }
        return null;
    }
}

