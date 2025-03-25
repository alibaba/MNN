// Created by ruoyi.sjd on 2025/1/9.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.utils;

import android.media.MediaPlayer;
import android.os.Handler;
import android.os.Looper;

public class AudioPlayService {

    public static final String TAG = "AudioPlayService";

    private static AudioPlayService instance;
    private MediaPlayer mediaPlayer;
    private String currentAudioPath = null;
    private Handler handler;
    private AudioPlayerCallback callback;

    // Private constructor to ensure singleton pattern
    private AudioPlayService() {
        handler = new Handler(Looper.getMainLooper());
    }

    // Singleton instance
    public static AudioPlayService getInstance() {
        if (instance == null) {
            synchronized (AudioPlayService.class) {
                if (instance == null) {
                    instance = new AudioPlayService();
                }
            }
        }
        return instance;
    }

    private boolean isMediaPlaying() {
        try {
            return mediaPlayer != null && mediaPlayer.isPlaying();
        } catch (IllegalStateException e) {
            return false;
        }
    }
    public void playAudio(String audioPath, AudioPlayerCallback callback) {
        if (audioPath == null) {
            return;
        }
        if (isMediaPlaying()) {
            mediaPlayer.stop();
            mediaPlayer.reset();
        }
        this.callback = callback;
        this.currentAudioPath = audioPath;

        mediaPlayer = new MediaPlayer();
        try {
            mediaPlayer.setDataSource(audioPath);
            mediaPlayer.prepare();
            mediaPlayer.start();

            if (callback != null) {
                callback.onPlayStart();
            }

            // Monitor progress
            monitorProgress();

            mediaPlayer.setOnCompletionListener(mp -> {
                if (callback != null) {
                    callback.onPlayFinish();
                }
                resetMediaPlayer();
            });

            mediaPlayer.setOnErrorListener((mp, what, extra) -> {
                if (callback != null) {
                    callback.onPlayError();
                }
                resetMediaPlayer();
                return true;
            });

        } catch (Exception e) {
            if (callback != null) {
                callback.onPlayError();
            }
            resetMediaPlayer();
        }
    }

    public void pauseAudio(String audioPath) {
        try {
            if (mediaPlayer != null && mediaPlayer.isPlaying()) {
                mediaPlayer.pause();
            }
        } catch (IllegalStateException e) {
            // Ignore the exception
        }
    }

    private void resetMediaPlayer() {
        if (mediaPlayer != null) {
            mediaPlayer.release();
            mediaPlayer = null;
            currentAudioPath = null;
        }
    }

    private void monitorProgress() {
        if (mediaPlayer != null && mediaPlayer.isPlaying()) {
            handler.postDelayed(() -> {
                if (mediaPlayer != null && mediaPlayer.isPlaying()) {
                    int currentPosition = mediaPlayer.getCurrentPosition();
                    int duration = mediaPlayer.getDuration();
                    float progress = (float) currentPosition / duration;
                    if (callback != null) {
                        callback.onPlayProgress(progress);
                    }
                    monitorProgress();
                }
            }, 200);
        }
    }

    public void seekAudio(String audioPath, float v) {
        if (!audioPath.equals(currentAudioPath)) {
            return;
        }
        if (mediaPlayer != null && mediaPlayer.isPlaying()) {
            mediaPlayer.seekTo((int) (v * mediaPlayer.getDuration()));
        }
    }

    public void destroy() {
        if (mediaPlayer != null) {
            mediaPlayer.release();
        }
        handler.removeMessages(0);
    }

    public interface AudioPlayerCallback {
        void onPlayFinish();
        void onPlayError();
        void onPlayStart();
        void onPlayProgress(float progress); // progress as percent
    }
}
