// Created by ruoyi.sjd on 2025/1/14.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.widgets

import android.content.Context
import android.util.AttributeSet
import android.util.TypedValue
import android.view.Gravity
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.TextView
import androidx.core.content.ContextCompat
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.utils.UiUtils.getThemeColor
import kotlin.math.max


class TagsLayout @JvmOverloads constructor(
    context: Context?,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) :
    LinearLayout(context, attrs, defStyleAttr) {
    private val tagMargin = resources.getDimensionPixelSize(R.dimen.tag_margin)
    private val tagPaddingHorizontal =
        resources.getDimensionPixelSize(R.dimen.tag_padding_horizontal)
    private val tagPaddingVertical = resources.getDimensionPixelSize(R.dimen.tag_padding_vertical)
    private val tagCornerRadius = resources.getDimension(R.dimen.tag_corner_radius)
    private var flexWrap = FLEX_WRAP_WRAP


    init {
        orientation = HORIZONTAL
        gravity = Gravity.START or Gravity.TOP
    }

    fun setFlexWrap(flexWrap: Int) {
        this.flexWrap = flexWrap
        updateLayout()
    }

    fun setTags(tags: List<String>) {
        removeAllViews()
        for (tagText in tags) {
            addTagView(tagText)
        }
        updateLayout()
    }

    private fun addTagView(tagText: String) {
        val tagView = TextView(context)
        tagView.text = tagText
        tagView.setTextColor(context.getThemeColor(com.google.android.material.R.attr.colorPrimary))
        tagView.setPadding(
            tagPaddingHorizontal,
            tagPaddingVertical,
            tagPaddingHorizontal,
            tagPaddingVertical
        )
        tagView.gravity = Gravity.CENTER
        tagView.setTextSize(TypedValue.COMPLEX_UNIT_PX, context.resources.getDimension(R.dimen.h4))
        tagView.background = ContextCompat.getDrawable(context, R.drawable.shape_tag_view)

        val layoutParams = LayoutParams(
            ViewGroup.LayoutParams.WRAP_CONTENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        )
        layoutParams.setMargins(tagMargin, tagMargin, tagMargin, tagMargin)
        tagView.layoutParams = layoutParams
        addView(tagView)
    }

    override fun onLayout(changed: Boolean, l: Int, t: Int, r: Int, b: Int) {
        if (flexWrap == FLEX_WRAP_WRAP) {
            performFlexWrapLayout(l, t, r, b)
        } else {
            super.onLayout(changed, l, t, r, b)
        }
    }

    private fun performFlexWrapLayout(l: Int, t: Int, r: Int, b: Int) {
        val width = r - l
        var currentLeft = paddingLeft
        var currentTop = paddingTop
        var maxHeight = 0

        for (i in 0 until childCount) {
            val child = getChildAt(i)
            if (child.visibility != GONE) {
                val lp = child.layoutParams as MarginLayoutParams
                val childWidth = child.measuredWidth + lp.leftMargin + lp.rightMargin
                val childHeight = child.measuredHeight + lp.topMargin + lp.bottomMargin

                if (currentLeft + childWidth > width - paddingRight) {
                    currentLeft = paddingLeft
                    currentTop += maxHeight
                    maxHeight = 0
                }

                child.layout(
                    currentLeft + lp.leftMargin,
                    currentTop + lp.topMargin,
                    currentLeft + lp.leftMargin + child.measuredWidth,
                    currentTop + lp.topMargin + child.measuredHeight
                )

                currentLeft += childWidth
                maxHeight = max(maxHeight.toDouble(), childHeight.toDouble()).toInt()
            }
        }
    }

    private fun updateLayout() {
        measure(MeasureSpec.UNSPECIFIED, MeasureSpec.UNSPECIFIED)
        requestLayout()
    }

    companion object {
        const val FLEX_WRAP_WRAP: Int = 1
        const val FLEX_WRAP_NOWRAP: Int = 0
    }
}