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
        if (imageUri == null) return
        showImagePopup(context, listOf(imageUri), 0)
    }

    fun showImagePopup(context: Context, images: List<Uri>, initialIndex: Int) {
        if (images.isEmpty()) return
        
        val inflater = context.getSystemService(Context.LAYOUT_INFLATER_SERVICE) as LayoutInflater
        val popupView = inflater.inflate(R.layout.popup_image_dialog, null)
        val dialog = Dialog(context, android.R.style.Theme_Black_NoTitleBar_Fullscreen)
        dialog.requestWindowFeature(Window.FEATURE_NO_TITLE)
        dialog.setContentView(popupView)
        
        val viewPager = popupView.findViewById<androidx.viewpager2.widget.ViewPager2>(R.id.viewPager)
        val adapter = FullScreenImageAdapter(images) {
            dialog.dismiss()
        }
        viewPager.adapter = adapter
        viewPager.setCurrentItem(initialIndex, false)
        
        popupView.setOnClickListener {
            dialog.dismiss()
        }
        
        dialog.show()
    }
}
