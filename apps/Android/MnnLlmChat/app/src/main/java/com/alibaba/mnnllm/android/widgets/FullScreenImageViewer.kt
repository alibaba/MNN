// Created by ruoyi.sjd on 2025/1/15.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.widgets

import android.app.Dialog
import android.content.Context
import android.net.Uri
import android.view.LayoutInflater
import android.view.ViewGroup
import android.view.Window
import android.widget.FrameLayout
import android.widget.ImageView
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.utils.UiUtils
import kotlin.math.min

object FullScreenImageViewer {
    fun showImagePopup(context: Context, imageUri: Uri?) {
        val inflater = context.getSystemService(Context.LAYOUT_INFLATER_SERVICE) as LayoutInflater
        val popupView = inflater.inflate(R.layout.popup_image_dialog, null)
        popupView.layoutParams = ViewGroup.LayoutParams(1000, 1000)
        val dialog = Dialog(context, R.style.Theme_TransparentFullScreenDialog)
        dialog.requestWindowFeature(Window.FEATURE_NO_TITLE)
        dialog.setContentView(popupView)
        val popupImageView = popupView.findViewById<ImageView>(R.id.popupImageView)
        popupImageView.setImageURI(imageUri)
        val size = UiUtils.getWindowSize(context)
        val dimension = min(size.x.toDouble(), size.y.toDouble()).toInt()
        popupImageView.layoutParams = FrameLayout.LayoutParams(dimension, dimension)
        dialog.show()
    }
}
