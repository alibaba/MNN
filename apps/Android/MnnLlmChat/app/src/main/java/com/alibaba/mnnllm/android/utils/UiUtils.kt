// Created by ruoyi.sjd on 2025/1/15.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.utils

import android.app.Activity
import android.content.Context
import android.content.ContextWrapper
import android.graphics.Point
import android.os.Handler
import android.os.Looper
import android.util.TypedValue
import android.widget.Toast

object UiUtils {
    fun getWindowSize(context: Context?): Point {
        val activity = getActivity(context)
        val display = activity!!.windowManager.defaultDisplay
        val size = Point()
        display.getSize(size)
        val width = size.x
        val height = size.y
        return Point(width, height)
    }

    fun getActivity(context: Context?): Activity? {
        if (context == null) {
            return null
        }
        if (context is Activity) {
            return context
        }
        if (context is ContextWrapper) {
            return getActivity(context.baseContext)
        }
        return null
    }

    private val uiHandler = Handler(Looper.getMainLooper())

    @JvmOverloads
    fun showToast(context: Context?, message: String?, duration: Int = Toast.LENGTH_SHORT) {
        if (Looper.myLooper() == Looper.getMainLooper()) {
            Toast.makeText(context, message, duration).show()
        } else {
            uiHandler.post { Toast.makeText(context, message, duration).show() }
        }
    }

    fun Context.getThemeColor(attrResId: Int): Int {
        val typedValue = TypedValue()
        theme.resolveAttribute(attrResId, typedValue, true)
        return typedValue.data
    }

}
