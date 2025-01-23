// Created by ruoyi.sjd on 2025/1/9.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.chat;

import android.widget.SeekBar;

import com.alibaba.mnnllm.android.R;
import com.alibaba.mnnllm.android.utils.AudioPlayService;

public class AudioPlayerComponent {

    private boolean isPlaying = false;
    private final AudioPlayService audioPlayService;
    private float playProgress;
    private final ChatDataItem chatDataItem;

    public final String TAG = "AudioPlayerComponent";
    private ChatViewHolders.UserViewHolder viewHolder;

    public AudioPlayerComponent(ChatDataItem chatDataItem) {
        this.chatDataItem = chatDataItem;
        this.audioPlayService = AudioPlayService.getInstance();
    }

    public void onPlayPauseClicked() {
        if (isPlaying) {
            audioPlayService.pauseAudio(this.chatDataItem.getAudioPath());
            setPlaying(false);
        } else {
            audioPlayService.playAudio(this.chatDataItem.getAudioPath(), new AudioPlayService.AudioPlayerCallback() {
                @Override
                public void onPlayFinish() {
                    setPlaying(false);
                    setProgress(0.0f);
                }

                @Override
                public void onPlayError() {
                    setPlaying(false);
                    setProgress(0.0f);
                }

                @Override
                public void onPlayStart() {
                    setPlaying(true);
                    if (viewHolder != null) {
                        viewHolder.iconPlayPause.setImageResource(R.drawable.ic_audio_pause);
                    }
                }

                @Override
                public void onPlayProgress(float progress) {
                    setProgress(progress);
                }
            });
        }
    }

    private void setProgress(float progress) {
        this.playProgress = progress;
        if (viewHolder != null) {
            viewHolder.audioSeekBar.setProgress((int) (progress * 100));
        }
    }

    private void updateProgressView() {
        if (viewHolder != null) {
            viewHolder.audioSeekBar.setProgress((int) (playProgress * 100));
        }
    }

    private void updatePlayingState() {
        viewHolder.iconPlayPause.setImageResource(isPlaying ? R.drawable.ic_audio_pause : R.drawable.ic_audio_play);
    }

    private void setPlaying(boolean playing) {
        if (playing != isPlaying) {
            isPlaying = playing;
            updatePlayingState();
        }
    }

    private void setSeekbarListener() {
        this.viewHolder.audioSeekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                if (fromUser) {
                    audioPlayService.seekAudio(chatDataItem.getAudioPath(), progress / 100f);
                }
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {

            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {

            }
        });
    }


    public void bindViewHolder(ChatViewHolders.UserViewHolder viewHolder) {
        this.viewHolder = viewHolder;
        setSeekbarListener();
        updateProgressView();
        updatePlayingState();
    }

}
