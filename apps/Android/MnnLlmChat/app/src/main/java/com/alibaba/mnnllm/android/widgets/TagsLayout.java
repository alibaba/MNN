// Created by ruoyi.sjd on 2025/1/14.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.widgets;


import android.content.Context;
import android.graphics.drawable.GradientDrawable;
import android.util.AttributeSet;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import android.widget.TextView;

import androidx.core.content.ContextCompat;


import com.alibaba.mnnllm.android.R;

import java.util.List;

public class TagsLayout extends LinearLayout {

    private final int tagMargin;
    private final int tagPaddingHorizontal;
    private final int tagPaddingVertical;
    private final float tagCornerRadius;
    private int flexWrap = FLEX_WRAP_WRAP;

    public static final int FLEX_WRAP_WRAP = 1;
    public static final int FLEX_WRAP_NOWRAP = 0;

    public TagsLayout(Context context) {
        this(context, null);
    }

    public TagsLayout(Context context, AttributeSet attrs) {
        this(context, attrs, 0);
    }

    public TagsLayout(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);

        tagMargin = getResources().getDimensionPixelSize(R.dimen.tag_margin);
        tagPaddingHorizontal = getResources().getDimensionPixelSize(R.dimen.tag_padding_horizontal);
        tagPaddingVertical = getResources().getDimensionPixelSize(R.dimen.tag_padding_vertical);
        tagCornerRadius = getResources().getDimension(R.dimen.tag_corner_radius);

        setOrientation(HORIZONTAL);
        setGravity(Gravity.START | Gravity.TOP);
    }

    public void setFlexWrap(int flexWrap) {
        this.flexWrap = flexWrap;
        updateLayout();
    }

    public void setTags(List<String> tags) {
        removeAllViews();
        for (String tagText : tags) {
            addTagView(tagText);
        }
        updateLayout();
    }

    private void addTagView(String tagText) {
        TextView tagView = new TextView(getContext());
        tagView.setText(tagText);
        tagView.setTextColor(ContextCompat.getColor(getContext(), android.R.color.white));
        tagView.setPadding(tagPaddingHorizontal, tagPaddingVertical, tagPaddingHorizontal, tagPaddingVertical);
        tagView.setGravity(Gravity.CENTER);
        tagView.setTextSize(TypedValue.COMPLEX_UNIT_PX, getContext().getResources().getDimension(R.dimen.h4));
        int backgroundColor = 0xFF666666;
        GradientDrawable background = new GradientDrawable();
        background.setCornerRadius(tagCornerRadius);
        background.setColor(backgroundColor);
        tagView.setBackground(background);

        LayoutParams layoutParams = new LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
        );
        layoutParams.setMargins(tagMargin, tagMargin, tagMargin, tagMargin);
        tagView.setLayoutParams(layoutParams);
        addView(tagView);
    }

    @Override
    protected void onLayout(boolean changed, int l, int t, int r, int b) {
        if (flexWrap == FLEX_WRAP_WRAP) {
            performFlexWrapLayout(l, t, r, b);
        } else {
            super.onLayout(changed, l, t, r, b);
        }
    }

    private void performFlexWrapLayout(int l, int t, int r, int b) {
        int width = r - l;
        int currentLeft = getPaddingLeft();
        int currentTop = getPaddingTop();
        int maxHeight = 0;

        for (int i = 0; i < getChildCount(); i++) {
            View child = getChildAt(i);
            if (child.getVisibility() != View.GONE) {
                MarginLayoutParams lp = (MarginLayoutParams) child.getLayoutParams();
                int childWidth = child.getMeasuredWidth() + lp.leftMargin + lp.rightMargin;
                int childHeight = child.getMeasuredHeight() + lp.topMargin + lp.bottomMargin;

                if (currentLeft + childWidth > width - getPaddingRight()) {
                    currentLeft = getPaddingLeft();
                    currentTop += maxHeight;
                    maxHeight = 0;
                }

                child.layout(
                        currentLeft + lp.leftMargin,
                        currentTop + lp.topMargin,
                        currentLeft + lp.leftMargin + child.getMeasuredWidth(),
                        currentTop + lp.topMargin + child.getMeasuredHeight()
                );

                currentLeft += childWidth;
                maxHeight = Math.max(maxHeight, childHeight);
            }
        }
    }

    private void updateLayout() {
        measure(MeasureSpec.UNSPECIFIED, MeasureSpec.UNSPECIFIED);
        requestLayout();
    }
}