// Created by ruoyi.sjd on 2025/3/5.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.utils

import android.content.Context
import android.view.LayoutInflater
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import com.alibaba.mnnllm.android.R
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import org.json.JSONObject
import org.json.JSONArray
import timber.log.Timber

object ConfigInfoDialog {
    fun show(context: Context, jsonContent: String) {
        val builder = MaterialAlertDialogBuilder(context)
        builder.setTitle(R.string.config_info)

        val dialogView = LayoutInflater.from(context).inflate(R.layout.dialog_config_viewer, null)
        val textView = dialogView.findViewById<TextView>(R.id.tv_json)
        val copyButton = dialogView.findViewById<Button>(R.id.btn_copy)

        val formattedJson = try {
            formatJson(jsonContent)
        } catch (e: Exception) {
            Timber.e(e, "Failed to format JSON config")
            jsonContent
        }
        
        textView.text = formattedJson

        copyButton.setOnClickListener {
            ClipboardUtils.copyToClipboard(context, jsonContent)
            Toast.makeText(context, R.string.copy_success, Toast.LENGTH_SHORT).show()
        }

        builder.setView(dialogView)
        builder.setPositiveButton(android.R.string.ok) { dialog, _ -> dialog.dismiss() }
        builder.create().show()
    }

    private fun formatJson(jsonString: String): String {
        return try {
            val trimmed = jsonString.trim()
            if (trimmed.startsWith("{")) {
                JSONObject(trimmed).toString(2)
            } else if (trimmed.startsWith("[")) {
                JSONArray(trimmed).toString(2)
            } else {
                jsonString
            }
        } catch (e: Exception) {
            jsonString
        }
    }
}
