package com.example.mnnllmchat;

import android.content.Context;
import android.util.AttributeSet;
import android.view.LayoutInflater;
import android.widget.SeekBar;
import android.widget.TextView;
import androidx.constraintlayout.widget.ConstraintLayout;
import com.google.android.material.materialswitch.MaterialSwitch;

public class SettingsRowSlideSwitch extends ConstraintLayout {

    private TextView labelSlider;
    private SeekBar seekBar;
    private MaterialSwitch switchSlider;

    public SettingsRowSlideSwitch(Context context, AttributeSet attrs) {
        super(context, attrs);
        init(context);
    }

    private void init(Context context) {
        LayoutInflater.from(context).inflate(R.layout.settings_row_slider_switch, this, true);

        labelSlider = findViewById(R.id.label_slider);
        seekBar = findViewById(R.id.seekbar);
        switchSlider = findViewById(R.id.switch_slider);
    }

    // Set label text
    public void setLabel(String label) {
        labelSlider.setText(label);
    }

    // Set seekbar value (0-1000 range)
    public void setSeekbarValue(int value) {
        seekBar.setProgress(value);
    }

    // Get current seekbar value
    public int getSeekbarValue() {
        return seekBar.getProgress();
    }

    // Set switch visibility
    public void setSwitchVisibility(int visibility) {
        switchSlider.setVisibility(visibility);
    }

    // Enable/disable the switch
    public void setSwitchEnabled(boolean enabled) {
        switchSlider.setEnabled(enabled);
    }
}