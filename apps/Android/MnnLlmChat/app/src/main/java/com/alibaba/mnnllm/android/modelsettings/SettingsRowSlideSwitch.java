// Created by ruoyi.sjd on 2025/4/29.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.modelsettings;

import android.content.Context;
import android.content.res.TypedArray;
import android.util.AttributeSet;
import android.view.LayoutInflater;
import android.widget.SeekBar;
import android.widget.TextView;
import android.view.View;
import androidx.constraintlayout.widget.ConstraintLayout;

import com.alibaba.mnnllm.android.R;
import com.google.android.material.materialswitch.MaterialSwitch;

public class SettingsRowSlideSwitch extends ConstraintLayout {

    private TextView labelSlider;
    private SeekBar seekBar;
    private MaterialSwitch switchSlider;

    public SettingsRowSlideSwitch(Context context, AttributeSet attrs) {
        super(context, attrs);
        init(context);
        TypedArray a = context.obtainStyledAttributes(attrs, R.styleable.SettingsRowSlideSwitch);
        String labelText = a.getString(R.styleable.SettingsRowSlideSwitch_labelText);
        int seekbarValue = a.getInt(R.styleable.SettingsRowSlideSwitch_seekbarValue, 0);
        int switchVisibility = a.getInt(R.styleable.SettingsRowSlideSwitch_switchVisibility, View.VISIBLE);
        boolean switchEnabled = a.getBoolean(R.styleable.SettingsRowSlideSwitch_switchEnabled, true);
        setLabel(labelText);
        setSeekbarValue(seekbarValue);
        setSwitchVisibility(switchVisibility);
        setSwitchEnabled(switchEnabled);
        a.recycle();
    }

    private void init(Context context) {
        LayoutInflater.from(context).inflate(R.layout.settings_row_slider_switch, this, true);

        labelSlider = findViewById(R.id.label_slider);
        seekBar = findViewById(R.id.seekbar);
        switchSlider = findViewById(R.id.switch_slider);
    }

    public void setLabel(String label) {
        labelSlider.setText(label);
    }

    public void setSeekbarValue(int value) {
        seekBar.setProgress(value);
    }

    public int getSeekbarValue() {
        return seekBar.getProgress();
    }

    public void setSwitchVisibility(int visibility) {
        switchSlider.setVisibility(visibility);
    }

    public void setSwitchEnabled(boolean enabled) {
        switchSlider.setEnabled(enabled);
    }
}