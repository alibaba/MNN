// Created by ruoyi.sjd on 2025/2/6.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.widgets;

import android.content.Context;
import android.graphics.Color;
import android.graphics.drawable.ColorDrawable;
import android.view.Gravity;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.PopupWindow;
import android.widget.TextView;

import com.alibaba.mnnllm.android.R;

public class PopupWindowHelper {

    public void showPopupWindow(Context context, View view, int x, int y, View.OnClickListener onClickListener) {
        // Inflate the popup_layout view
        View popupView = LayoutInflater.from(context).inflate(R.layout.assistant_text_popup_menu, null);

        PopupWindow popupWindow = new PopupWindow(popupView,
                ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
                true);

        // If you want to dismiss popup on outside touch
        popupWindow.setOutsideTouchable(true);
        popupWindow.setBackgroundDrawable(new ColorDrawable(Color.TRANSPARENT));

        TextView copyItem = popupView.findViewById(R.id.assistant_text_copy);
        TextView selectItem = popupView.findViewById(R.id.assistant_text_select);
        TextView reportIssueItem = popupView.findViewById(R.id.assistant_text_report);

        copyItem.setOnClickListener(v -> {
            onClickListener.onClick(v);
            popupWindow.dismiss();
        });

        selectItem.setOnClickListener(v -> {
            onClickListener.onClick(v);
            popupWindow.dismiss();
        });

        reportIssueItem.setOnClickListener(v -> {
            onClickListener.onClick(v);
            popupWindow.dismiss();
        });

        popupWindow.showAtLocation(view, Gravity.NO_GRAVITY, x, y);
    }
}
