// Created by ruoyi.sjd on 2025/3/5.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.utils

import android.content.Context
import android.view.LayoutInflater
import android.widget.Button
import android.widget.Toast
import com.alibaba.mnnllm.android.R
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.yuyh.jsonviewer.library.JsonRecyclerView
import timber.log.Timber

object ConfigInfoDialog {
    fun show(context: Context, jsonContent: String) {
        val builder = MaterialAlertDialogBuilder(context)
        builder.setTitle(R.string.config_info)

        val dialogView = LayoutInflater.from(context).inflate(R.layout.dialog_config_viewer, null)
        val recyclerView = dialogView.findViewById<JsonRecyclerView>(R.id.rv_json)
        val copyButton = dialogView.findViewById<Button>(R.id.btn_copy)

        try {
            recyclerView.bindJson(jsonContent)
        } catch (e: Exception) {
            Timber.e(e, "Failed to parse JSON config")
            recyclerView.bindJson("{\"error\": \"Failed to parse config\"}")
        }

        copyButton.setOnClickListener {
            ClipboardUtils.copyToClipboard(context, jsonContent)
            Toast.makeText(context, R.string.copy_success, Toast.LENGTH_SHORT).show()
        }

        builder.setView(dialogView)
        builder.setPositiveButton(android.R.string.ok) { dialog, _ -> dialog.dismiss() }
        builder.create().show()
    }
}
