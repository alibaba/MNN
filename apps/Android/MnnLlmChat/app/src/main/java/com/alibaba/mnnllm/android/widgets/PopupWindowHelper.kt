// Created by ruoyi.sjd on 2025/2/6.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.widgets

import android.content.Context
import android.graphics.Color
import android.graphics.drawable.ColorDrawable
import android.view.ContextThemeWrapper
import android.view.Gravity
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.PopupWindow
import android.widget.TextView
import com.alibaba.mnnllm.android.R

class PopupWindowHelper {
    data class AnchorBounds(
        val left: Int,
        val top: Int,
        val right: Int,
        val bottom: Int
    )

    data class PopupPosition(
        val x: Int,
        val y: Int
    )

    fun showPopupWindow(
        context: Context?,
        anchorView: View?,
        onClickListener: View.OnClickListener
    ) {
        if (context == null || anchorView == null) {
            return
        }

        val popupView = createPopupView(LayoutInflater.from(context), null, onClickListener)

        val popupWindow = PopupWindow(
            popupView,
            ViewGroup.LayoutParams.WRAP_CONTENT,
            ViewGroup.LayoutParams.WRAP_CONTENT,
            true
        )

        // If you want to dismiss popup on outside touch
        popupWindow.isOutsideTouchable = true
        popupWindow.setBackgroundDrawable(ColorDrawable(Color.TRANSPARENT))
        popupWindow.elevation = dpToPx(context, 12).toFloat()

        popupView.measure(
            View.MeasureSpec.makeMeasureSpec(
                context.resources.displayMetrics.widthPixels,
                View.MeasureSpec.AT_MOST
            ),
            View.MeasureSpec.makeMeasureSpec(0, View.MeasureSpec.UNSPECIFIED)
        )

        val location = IntArray(2)
        anchorView.getLocationOnScreen(location)
        val popupPosition = calculatePopupPosition(
            anchorBounds = AnchorBounds(
                left = location[0],
                top = location[1],
                right = location[0] + anchorView.width,
                bottom = location[1] + anchorView.height
            ),
            popupWidth = popupView.measuredWidth,
            popupHeight = popupView.measuredHeight,
            screenWidth = context.resources.displayMetrics.widthPixels,
            screenHeight = context.resources.displayMetrics.heightPixels,
            margin = dpToPx(context, SCREEN_MARGIN_DP),
            spacing = dpToPx(context, ANCHOR_SPACING_DP)
        )

        popupWindow.showAtLocation(anchorView.rootView, Gravity.NO_GRAVITY, popupPosition.x, popupPosition.y)
    }

    companion object {
        private const val SCREEN_MARGIN_DP = 12
        private const val ANCHOR_SPACING_DP = 8

        fun createPopupView(
            inflater: LayoutInflater,
            parent: ViewGroup?,
            onClickListener: View.OnClickListener? = null
        ): View {
            val themedInflater = inflater.cloneInContext(
                ContextThemeWrapper(inflater.context, R.style.AppTheme)
            )
            val popupView = themedInflater.inflate(R.layout.assistant_text_popup_menu, parent, false)
            if (onClickListener == null) {
                return popupView
            }

            val copyItem = popupView.findViewById<TextView>(R.id.assistant_text_copy)
            val selectItem = popupView.findViewById<TextView>(R.id.assistant_text_select)
            val reportIssueItem = popupView.findViewById<TextView>(R.id.assistant_text_report)

            wireAction(popupView.findViewById(R.id.assistant_text_copy_row), copyItem, onClickListener)
            wireAction(popupView.findViewById(R.id.assistant_text_copy_icon), copyItem, onClickListener)
            wireAction(copyItem, copyItem, onClickListener)

            wireAction(popupView.findViewById(R.id.assistant_text_select_row), selectItem, onClickListener)
            wireAction(popupView.findViewById(R.id.assistant_text_select_icon), selectItem, onClickListener)
            wireAction(selectItem, selectItem, onClickListener)

            wireAction(popupView.findViewById(R.id.assistant_text_report_row), reportIssueItem, onClickListener)
            wireAction(popupView.findViewById(R.id.assistant_text_report_icon), reportIssueItem, onClickListener)
            wireAction(reportIssueItem, reportIssueItem, onClickListener)

            return popupView
        }

        fun calculatePopupPosition(
            anchorBounds: AnchorBounds,
            popupWidth: Int,
            popupHeight: Int,
            screenWidth: Int,
            screenHeight: Int,
            margin: Int,
            spacing: Int
        ): PopupPosition {
            val maxX = (screenWidth - popupWidth - margin).coerceAtLeast(margin)
            val x = anchorBounds.left.coerceIn(margin, maxX)

            val belowY = anchorBounds.bottom + spacing
            val aboveY = anchorBounds.top - popupHeight - spacing
            val fitsBelow = belowY + popupHeight + margin <= screenHeight
            val fitsAbove = aboveY >= margin
            val maxY = (screenHeight - popupHeight - margin).coerceAtLeast(margin)

            val y = when {
                fitsBelow -> belowY
                fitsAbove -> aboveY
                (screenHeight - anchorBounds.bottom) >= anchorBounds.top -> belowY.coerceIn(margin, maxY)
                else -> aboveY.coerceIn(margin, maxY)
            }

            return PopupPosition(x = x, y = y)
        }

        private fun wireAction(
            triggerView: View,
            actionView: TextView,
            onClickListener: View.OnClickListener
        ) {
            triggerView.setOnClickListener { onClickListener.onClick(actionView) }
        }

        private fun dpToPx(context: Context, dp: Int): Int {
            return (dp * context.resources.displayMetrics.density).toInt()
        }
    }
}
