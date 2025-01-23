// Created by ruoyi.sjd on 2025/01/03.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.chat;

import android.annotation.SuppressLint;
import android.net.Uri;
import android.text.TextUtils;
import android.view.View;
import android.widget.ImageView;
import android.widget.SeekBar;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import com.alibaba.mnnllm.android.utils.ModelUtils;
import com.alibaba.mnnllm.android.R;
import com.alibaba.mnnllm.android.utils.PreferenceUtils;
import com.alibaba.mnnllm.android.widgets.FullScreenImageViewer;

public class ChatViewHolders {
    public static final int HEADER = 0, ASSISTANT = 1, USER = 2;

    public static class HeaderViewHolder extends RecyclerView.ViewHolder {

        private final TextView viewTime;
        public HeaderViewHolder(@NonNull View itemView) {
            super(itemView);
            viewTime = itemView.findViewById(R.id.tv_date);
        }

        public void bind(ChatDataItem data) {
            viewTime.setText(data.getTime());
        }
    }

    public static class UserViewHolder extends RecyclerView.ViewHolder implements View.OnClickListener {

        public  final View audioLayout;
        public final TextView viewText;

        public final ImageView chatImage;

        public final TextView textDuration;

        public final ImageView iconPlayPause;
        public final SeekBar audioSeekBar;

        public UserViewHolder(@NonNull View itemView) {
            super(itemView);
            viewText = itemView.findViewById(R.id.tv_chat_text);
            chatImage = itemView.findViewById(R.id.tv_chat_image);
            audioLayout = itemView.findViewById(R.id.layout_audio);
            iconPlayPause = itemView.findViewById(R.id.iv_audio_play_pause);
            textDuration = itemView.findViewById(R.id.tv_chat_voice_duration);
            audioSeekBar = itemView.findViewById(R.id.audio_seek_bar);
            iconPlayPause.setOnClickListener(this);
        }

        @SuppressLint("DefaultLocale")
        private static String formatTime(int seconds) {
            int minutes = seconds / 60;
            int remainingSeconds = seconds % 60;
            return String.format("%d:%02d", minutes, remainingSeconds);
        }

        @SuppressLint("DefaultLocale")
        public void bind(ChatDataItem data) {
            audioLayout.setVisibility(data.getAudioUri() != null ? View.VISIBLE : View.GONE);
            audioLayout.setTag(data);
            iconPlayPause.setTag(data);
            viewText.setText(data.getText());
            viewText.setVisibility(TextUtils.isEmpty(data.getText()) ? View.GONE : View.VISIBLE);
            textDuration.setText(formatTime((int) data.getAudioDuration()));
            Uri imageUri = data.getImageUri();
            chatImage.setVisibility(imageUri != null ? View.VISIBLE : View.GONE);
            if (imageUri != null) {
                chatImage.setImageURI(imageUri);
            }
            if (data.audioPlayComponent != null) {
                data.audioPlayComponent.bindViewHolder(this);
            }
        }

        @Override
        public void onClick(View v) {
            ChatDataItem chatDataItem = (ChatDataItem) v.getTag();
            if (chatDataItem.getAudioUri() != null) {
                if (chatDataItem.audioPlayComponent == null) {
                    chatDataItem.audioPlayComponent = new AudioPlayerComponent(chatDataItem);
                }
                chatDataItem.audioPlayComponent.bindViewHolder(this);
                chatDataItem.audioPlayComponent.onPlayPauseClicked();
            }
        }
    }


    public static class AssistantViewHolder extends RecyclerView.ViewHolder implements View.OnClickListener {

        static final String TAG = "AssistantViewHolder";
        private final TextView viewText;
        private final TextView benchmarkInfo;

        private final ImageView headerIcon;

        private final ImageView imageGenerated;
        public View viewAssistantLoading;


        public AssistantViewHolder(@NonNull View view) {
            super(view);
            viewText = view.findViewById(R.id.tv_chat_text);
            headerIcon = view.findViewById(R.id.ic_header);
            viewAssistantLoading = view.findViewById(R.id.view_assistant_loading);
            benchmarkInfo = view.findViewById(R.id.tv_chat_benchmark);
            imageGenerated = view.findViewById(R.id.image_generated);
            imageGenerated.setOnClickListener(this);
        }

        public void bind(ChatDataItem data, String modelName) {
            if (TextUtils.isEmpty(data.getText())) {
                viewAssistantLoading.setVisibility(View.VISIBLE);
                viewText.setVisibility(View.GONE);
            } else {
                viewText.setVisibility(View.VISIBLE);
                viewText.setText(data.getText());
                viewAssistantLoading.setVisibility(View.GONE);
            }
            boolean showMetrics = PreferenceUtils.getBoolean(itemView.getContext(), PreferenceUtils.KEY_SHOW_PERFORMACE_METRICS, true);
            if (showMetrics && !TextUtils.isEmpty(data.getBenchmarkInfo())) {
                benchmarkInfo.setVisibility(View.VISIBLE);
                benchmarkInfo.setText(data.getBenchmarkInfo());
            } else {
                benchmarkInfo.setVisibility(View.GONE);
            }
            imageGenerated.setVisibility(data.getImageUri() != null ? View.VISIBLE : View.GONE);
            if (data.getImageUri() != null) {
                imageGenerated.setImageURI(data.getImageUri());
            }
            int drawableId = ModelUtils.getDrawableId(modelName);
            headerIcon.setImageResource(drawableId > 0 ? drawableId : R.drawable.ic_launcher);
            imageGenerated.setTag(data);
        }

        @Override
        public void onClick(View v) {
            ChatDataItem data = (ChatDataItem) v.getTag();
            FullScreenImageViewer.showImagePopup(v.getContext(), data.getImageUri());
        }
    }
}
