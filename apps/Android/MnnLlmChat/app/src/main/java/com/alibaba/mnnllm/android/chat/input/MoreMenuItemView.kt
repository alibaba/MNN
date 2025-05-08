// Created by ruoyi.sjd on 2025/1/9.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.chat.input

import android.content.Context
import android.util.AttributeSet
import android.widget.ImageView
import android.widget.RelativeLayout
import android.widget.TextView
import com.alibaba.mnnllm.android.R

class MoreMenuItemView @JvmOverloads constructor(context: Context, attrs: AttributeSet? = null) :
    RelativeLayout(context, attrs) {
    init {
        init(context, attrs)
    }

    override fun onFinishInflate() {
        super.onFinishInflate()
    }

    private fun init(context: Context, attrs: AttributeSet?) {
        inflate(context, R.layout.chat_more_menu_item, this)
        val imageView = findViewById<ImageView>(R.id.icon)
        val textView = findViewById<TextView>(R.id.text)
        if (attrs != null) {
            val a = context.theme.obtainStyledAttributes(
                attrs,
                R.styleable.MoreMenuItemView,
                0, 0
            )

            try {
                val text = a.getString(R.styleable.MoreMenuItemView_text)
                val icon = a.getDrawable(R.styleable.MoreMenuItemView_icon)
                if (text != null) {
                    textView.text = text
                }
                if (icon != null) {
                    imageView.setImageDrawable(icon)
                }
            } finally {
                a.recycle()
            }
        }
    }
}
