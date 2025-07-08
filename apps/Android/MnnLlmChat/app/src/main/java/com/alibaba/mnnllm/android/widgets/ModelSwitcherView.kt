
// Created by ruoyi.sjd on 2025/6/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.widgets

import android.content.Context
import android.util.AttributeSet
import android.view.LayoutInflater
import android.widget.LinearLayout
import android.widget.TextView
import com.alibaba.mnnllm.android.R

class ModelSwitcherView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : LinearLayout(context, attrs, defStyleAttr) {

    private val selectedModelTextView: TextView

    init {
        orientation = HORIZONTAL
        LayoutInflater.from(context).inflate(R.layout.view_model_switcher, this, true)
        selectedModelTextView = findViewById(R.id.tv_selected_model)
    }

    var text : String
        get() = selectedModelTextView.text.toString()
        set(value) {
            selectedModelTextView.text = value
        }
}