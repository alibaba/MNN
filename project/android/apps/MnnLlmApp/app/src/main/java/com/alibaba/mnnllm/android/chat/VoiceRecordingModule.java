// Created by ruoyi.sjd on 2025/01/08.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.chat;

import static android.view.MotionEvent.ACTION_UP;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.graphics.Rect;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.alibaba.mnnllm.android.R;
import com.alibaba.mnnllm.android.utils.FileUtils;
import com.github.squti.androidwaverecorder.WaveRecorder;

import java.io.File;

public class VoiceRecordingModule {

    private View buttonVoiceRecording;
    private ImageView buttonSwitchVoice;

    private final ChatActivity activity;
    private VoiceRecordingListener listener;
    private WaveRecorder waveRecorder;;
    private View voceRecordingWave;

    private TextView textVoiceHint;

    private String recordingFilePath;

    private long mStartRecordTime;

    public static final int REQUEST_RECORD_AUDIO_PERMISSION = 999;

    public static final String TAG = "VoiceRecordingModule";
    private boolean isCancelRecord;
    private boolean enabled = false;

    public VoiceRecordingModule(ChatActivity activity) {
        this.activity = activity;
    }

    public void setOnVoiceRecordingListener(VoiceRecordingListener listener) {
        this.listener = listener;
    }

    public void setup(boolean isAudioModel) {
        buttonSwitchVoice = activity.findViewById(R.id.bt_switch_audio);
        if (!isAudioModel) {
            buttonSwitchVoice.setVisibility(View.GONE);
            return;
        }
        voceRecordingWave = activity.findViewById(R.id.voice_recording_wav);
        buttonVoiceRecording = activity.findViewById(R.id.btn_voice_recording);
        textVoiceHint = activity.findViewById(R.id.text_voice_hint);
        buttonSwitchVoice.setOnClickListener(v -> handleSwitch());
        buttonVoiceRecording.setOnTouchListener(this::handleTouchEvent);
        handleSwitch();
    }

    private boolean handleTouchEvent(View v, MotionEvent event) {
        if (!this.enabled) {
            return false;
        }
        if (event.getAction() == ACTION_UP) {
            Log.d(TAG, "onTouch up stop recording");
            if (waveRecorder == null) {
                return false;
            }
            endAudioRecording(isCancelRecord);
        } else if (event.getAction() == MotionEvent.ACTION_DOWN) {

            if (ContextCompat.checkSelfPermission(activity, android.Manifest.permission.RECORD_AUDIO)
                    != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(activity,
                        new String[]{Manifest.permission.RECORD_AUDIO},
                        REQUEST_RECORD_AUDIO_PERMISSION);
            } else {
                startAudioRecording();
            }

        } else if (event.getAction() == MotionEvent.ACTION_MOVE) {
            if (!isPointInsideView(event.getRawX(), event.getRawY(), v)) {
                isCancelRecord = true;
                updateRecordingUI();
            } else {
                isCancelRecord = false;
                updateRecordingUI();
            }
        } else if (event.getAction() == MotionEvent.ACTION_CANCEL) {
            if (waveRecorder != null) {
                endAudioRecording(true);
            }
        }
        return true;
    }

    private void updateRecordingUI() {
        textVoiceHint.setVisibility(View.VISIBLE);
        textVoiceHint.setText(isCancelRecord ? R.string.release_to_cancel : R.string.release_to_send);
        voceRecordingWave.setBackgroundColor(isCancelRecord ? Color.RED : voceRecordingWave.getResources().getColor(R.color.colorAccent));
        textVoiceHint.setTextColor(isCancelRecord ? Color.RED : Color.BLACK);
    }

    private boolean isPointInsideView(float x, float y, View view) {
        int[] location = new int[2];
        view.getLocationOnScreen(location);
        Rect rect = new Rect(
                location[0],
                location[1],
                location[0] + view.getWidth(),
                location[1] + view.getHeight()
        );
        return rect.contains((int) x, (int) y);
    }

    private void endAudioRecording(boolean cancel) {
        waveRecorder.stopRecording();
        voceRecordingWave.setVisibility(View.GONE);
        if (listener != null) {
            float duration = (System.currentTimeMillis() - mStartRecordTime) / 1000f;
            if (!cancel && duration > 1) {
                listener.onRecordSuccess(duration, this.recordingFilePath);
            } else {
                new File(this.recordingFilePath).delete();
                listener.onRecordCanceled();
            }
        }
        textVoiceHint.setVisibility(View.GONE);
        waveRecorder = null;
        isCancelRecord = false;
    }

    private void startAudioRecording() {
        recordingFilePath = FileUtils.generateDestRecordFilePath(activity, activity.getSessionId());
        Log.d(TAG, "onTouch down start recording: " + recordingFilePath);
        waveRecorder = new WaveRecorder(recordingFilePath);
        mStartRecordTime = System.currentTimeMillis();
        waveRecorder.startRecording();
        voceRecordingWave.setVisibility(View.VISIBLE);
        updateRecordingUI();
    }

    private void handleSwitch() {
        if (buttonVoiceRecording.getVisibility() == View.VISIBLE) {
            exitRecordingMode();
        } else {
            enterRecordingMode();
        }
    }

    public void enterRecordingMode() {
        if (buttonVoiceRecording.getVisibility() == View.VISIBLE) {
            return;
        }
        buttonVoiceRecording.setVisibility(View.VISIBLE);
        if (listener != null) {
            listener.onEnterRecordingMode();
        }
        buttonSwitchVoice.setImageResource(R.drawable.ic_keyboard);
    }

    public void exitRecordingMode() {
        if (buttonVoiceRecording == null || buttonVoiceRecording.getVisibility() != View.VISIBLE) {
            return;
        }
        buttonVoiceRecording.setVisibility(View.GONE);
        if (listener != null) {
            listener.onLeaveRecordingMode();
        }
        buttonSwitchVoice.setImageResource(R.drawable.ic_audio);
    }

    public void handlePermissionAllowed() {

    }

    public void handlePermissionDenied() {
        Toast.makeText(this.activity, R.string.recording_permission_denied, Toast.LENGTH_LONG).show();
    }

    public void onEnabled() {
        this.enabled = true;
    }

    public interface VoiceRecordingListener {
        void onEnterRecordingMode();
        void onLeaveRecordingMode();

        void onRecordSuccess(float duration, String recordingFilePath);

        void onRecordCanceled();
    }
}
