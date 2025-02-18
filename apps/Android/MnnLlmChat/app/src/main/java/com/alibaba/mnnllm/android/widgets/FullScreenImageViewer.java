// Created by ruoyi.sjd on 2025/1/15.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.widgets;

import static android.content.Context.LAYOUT_INFLATER_SERVICE;

import android.app.Dialog;
import android.content.Context;
import android.graphics.Point;
import android.net.Uri;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.view.Window;
import android.widget.FrameLayout;
import android.widget.ImageView;

import com.alibaba.mnnllm.android.R;
import com.alibaba.mnnllm.android.utils.UiUtils;

public class FullScreenImageViewer {
    public static void showImagePopup(Context context, Uri imageUri) {
        LayoutInflater inflater = (LayoutInflater) context.getSystemService(LAYOUT_INFLATER_SERVICE);
        View popupView = inflater.inflate(R.layout.popup_image_dialog, null);
        popupView.setLayoutParams(new ViewGroup.LayoutParams(1000, 1000));
        Dialog dialog = new Dialog(context, R.style.Theme_TransparentFullScreenDialog);
        dialog.requestWindowFeature(Window.FEATURE_NO_TITLE);
        dialog.setContentView(popupView);
        ImageView popupImageView = popupView.findViewById(R.id.popupImageView);
        popupImageView.setImageURI(imageUri);
        Point size = UiUtils.getWindowSize(context);
        int dimension = Math.min(size.x, size.y);
        popupImageView.setLayoutParams(new FrameLayout.LayoutParams(dimension, dimension));
        dialog.show();
    }
}
