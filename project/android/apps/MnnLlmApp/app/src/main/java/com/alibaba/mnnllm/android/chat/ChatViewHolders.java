// Created by ruoyi.sjd on 2025/01/03.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.chat;

import android.annotation.SuppressLint;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Context;
import android.content.Intent;
import android.net.Uri;
import android.text.TextUtils;
import android.view.MenuInflater;
import android.view.MotionEvent;
import android.view.View;
import android.widget.ImageView;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.widget.PopupMenu;
import androidx.recyclerview.widget.RecyclerView;

import com.alibaba.mls.api.HfRepoItem;
import com.alibaba.mls.api.download.ModelDownloadManager;
import com.alibaba.mnnllm.android.utils.GithubUtils;
import com.alibaba.mnnllm.android.utils.ModelUtils;
import com.alibaba.mnnllm.android.R;
import com.alibaba.mnnllm.android.utils.PreferenceUtils;
import com.alibaba.mnnllm.android.widgets.FullScreenImageViewer;
import com.alibaba.mnnllm.android.widgets.PopupWindowHelper;

import io.noties.markwon.Markwon;

import java.util.List;

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


    public static class AssistantViewHolder extends RecyclerView.ViewHolder implements View.OnClickListener, View.OnLongClickListener {

        static final String TAG = "AssistantViewHolder";
        private final TextView viewText;
        private final TextView benchmarkInfo;

        private final ImageView headerIcon;

        private final ImageView imageGenerated;
        private final Markwon markdown;
        public View viewAssistantLoading;

        private int lastTouchX = 0;
        private int lastTouchY = 0;

        @SuppressLint("ClickableViewAccessibility")
        public AssistantViewHolder(@NonNull View view) {
            super(view);
            markdown = Markwon.create(itemView.getContext());
            viewText = view.findViewById(R.id.tv_chat_text);
            headerIcon = view.findViewById(R.id.ic_header);
            viewAssistantLoading = view.findViewById(R.id.view_assistant_loading);
            benchmarkInfo = view.findViewById(R.id.tv_chat_benchmark);
            imageGenerated = view.findViewById(R.id.image_generated);
            viewText.setOnLongClickListener(this);
            viewText.setOnTouchListener(new View.OnTouchListener() {
                @SuppressLint("ClickableViewAccessibility")
                @Override
                public boolean onTouch(View v, MotionEvent event) {
                    if (event.getAction() == MotionEvent.ACTION_DOWN) {
                        int[] location = new int[2];
                        v.getLocationOnScreen(location);
                        lastTouchX = location[0] + (int) event.getX();
                        lastTouchY = location[1] + (int) event.getY();
                    }
                    return false;
                }
            });
            imageGenerated.setOnClickListener(this);
        }

        public void bind(ChatDataItem data, String modelName, List<Object> payloads) {
            if (payloads != null && !payloads.isEmpty()) {
                markdown.setMarkdown(viewText, data.getDisplayText());
                return;
            }
            if (TextUtils.isEmpty(data.getText())) {
                viewAssistantLoading.setVisibility(View.VISIBLE);
                viewText.setVisibility(View.GONE);
            } else {
                markdown.setMarkdown(viewText, data.getDisplayText());
                viewText.setVisibility(View.VISIBLE);
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
            viewText.setTag(data);
        }

        @Override
        public void onClick(View v) {
            ChatDataItem data = (ChatDataItem) v.getTag();
            FullScreenImageViewer.showImagePopup(v.getContext(), data.getImageUri());
        }

        @Override
        public boolean onLongClick(View v) {
            TextView textView = (TextView) v;
            ChatDataItem chatDataItem = (ChatDataItem) v.getTag();
            new PopupWindowHelper().showPopupWindow(v.getContext(), v, this.lastTouchX, this.lastTouchY, new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                if (v.getId() == R.id.assistant_text_copy) {
                    copyText(itemView.getContext(), textView);
                } else if (v.getId() == R.id.assistant_text_select) {
                    Intent intent = new Intent(v.getContext(), SelectTextActivity.class);
                    intent.putExtra("content", chatDataItem.getText());
                    v.getContext().startActivity(intent);
                } else if (v.getId() == R.id.assistant_text_report) {
                    GithubUtils.reportIssue(v.getContext());
                }
                }
            });
            return true;
        }

        private void copyText(Context context, TextView textView) {
            String content = textView.getText().toString();
            ClipboardManager clipboard = (ClipboardManager) context.getSystemService(Context.CLIPBOARD_SERVICE);
            ClipData clip = ClipData.newPlainText("CopiedText", content);
            clipboard.setPrimaryClip(clip);
            Toast.makeText(context, R.string.copy_success, Toast.LENGTH_SHORT).show();
        }
    }
}
