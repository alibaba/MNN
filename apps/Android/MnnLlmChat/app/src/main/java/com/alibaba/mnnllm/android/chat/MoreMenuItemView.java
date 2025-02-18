// Created by ruoyi.sjd on 2025/1/9.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.chat;

import android.content.Context;
import android.content.res.TypedArray;
import android.graphics.drawable.Drawable;
import android.util.AttributeSet;
import android.widget.ImageView;
import android.widget.RelativeLayout;
import android.widget.TextView;

import com.alibaba.mnnllm.android.R;

public class MoreMenuItemView extends RelativeLayout {

    public MoreMenuItemView(Context context) {
        this(context, null);
    }

    public MoreMenuItemView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init(context, attrs);
    }

    @Override
    protected void onFinishInflate() {
        super.onFinishInflate();
    }

    private void init(Context context, AttributeSet attrs) {
        inflate(context, R.layout.chat_more_menu_item, this);
        ImageView imageView  = findViewById(R.id.icon);
        TextView textView = findViewById(R.id.text);
        if (attrs != null) {
            TypedArray a = context.getTheme().obtainStyledAttributes(
                    attrs,
                    R.styleable.MoreMenuItemView,
                    0, 0);

            try {
                String text = a.getString(R.styleable.MoreMenuItemView_text);
                Drawable icon = a.getDrawable(R.styleable.MoreMenuItemView_icon);
                if (text != null) {
                    textView.setText(text);
                }
                if (icon != null) {
                    imageView.setImageDrawable(icon);
                }

            } finally {
                a.recycle();
            }
        }
    }
}
