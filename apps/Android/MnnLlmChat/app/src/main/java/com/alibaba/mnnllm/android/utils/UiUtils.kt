// Created by ruoyi.sjd on 2025/1/15.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.utils

import android.app.Activity
import android.app.ActivityManager
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.ContextWrapper
import android.graphics.Point
import android.os.Handler
import android.os.Looper
import android.util.TypedValue
import android.widget.TextView
import android.widget.Toast
import androidx.core.content.ContextCompat
import com.alibaba.mnnllm.android.R
import kotlin.math.roundToInt

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
        // An attribute pointing at a colour resource resolves to a reference, where `data` holds
        // the resource id instead of an ARGB value.
        return if (typedValue.resourceId != 0) {
            ContextCompat.getColor(this, typedValue.resourceId)
        } else {
            typedValue.data
        }
    }

    fun Context.dpToPx(dp: Int): Int {
        return TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP,
            dp.toFloat(),
            resources.displayMetrics
        ).roundToInt()
    }

    fun copyText(context: Context, textView: TextView) {
        val content = textView.text.toString()
        val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        val clip = ClipData.newPlainText("CopiedText", content)
        clipboard.setPrimaryClip(clip)
        Toast.makeText(context, R.string.copy_success, Toast.LENGTH_SHORT).show()
    }

    /**
     * 获取当前最上层的 Activity
     * @return 最上层 Activity，如果无法获取则返回 null
     */
    fun getTopActivity(): Activity? {
        return CurrentActivityTracker.currentActivity
    }
}