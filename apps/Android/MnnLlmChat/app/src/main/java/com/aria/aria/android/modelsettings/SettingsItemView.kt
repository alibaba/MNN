// Created by ruoyi.sjd on 2025/4/30.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.modelsettings

import android.content.Context
import android.util.AttributeSet
import android.view.LayoutInflater
import androidx.constraintlayout.widget.ConstraintLayout
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.databinding.ViewSettingsItemBinding

class SettingsItemView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : ConstraintLayout(context, attrs, defStyleAttr) {

    private val binding: ViewSettingsItemBinding =
        ViewSettingsItemBinding.inflate(
            LayoutInflater.from(context),
            this,
        )

    private var listener: ((Boolean) -> Unit)? = null

    init {
        attrs?.let {
            val ta = context.obtainStyledAttributes(it, R.styleable.SettingsItemView, 0, 0)
            binding.tvLabel.text = ta.getString(R.styleable.SettingsItemView_itemText) ?: ""
            binding.switchControl.isChecked = ta.getBoolean(R.styleable.SettingsItemView_checked, false)
            ta.recycle()
        }

        setOnClickListener {
            binding.switchControl.toggle()
            listener?.invoke(binding.switchControl.isChecked)
        }
        binding.switchControl.setOnCheckedChangeListener { _, checked ->
            listener?.invoke(checked)
        }
    }

    fun setItemText(text: String) {
        binding.tvLabel.text = text
    }

    fun setOnCheckedChangeListener(l: (Boolean) -> Unit) {
        listener = l
    }

    var isChecked: Boolean
        get() = binding.switchControl.isChecked
        set(v) { binding.switchControl.isChecked = v }
}

