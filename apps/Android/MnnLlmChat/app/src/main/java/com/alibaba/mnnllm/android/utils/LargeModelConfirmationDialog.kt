// Created by ruoyi.sjd on 2025/1/13.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.utils

import android.content.Context
import androidx.fragment.app.Fragment
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.alibaba.mnnllm.android.R

object LargeModelConfirmationDialog {
    
    /**
     * Show confirmation dialog for large models (>7GB)
     * @param context Context to show dialog
     * @param modelName Name of the model
     * @param modelSize Size of the model in GB
     * @param onConfirm Callback when user confirms to continue
     * @param onCancel Callback when user cancels (optional)
     */
    fun show(
        context: Context,
        modelName: String,
        modelSize: Double,
        onConfirm: () -> Unit,
        onCancel: (() -> Unit)? = null
    ) {
        MaterialAlertDialogBuilder(context, com.google.android.material.R.style.ThemeOverlay_Material3_MaterialAlertDialog)
            .setTitle(context.getString(R.string.large_model_warning_title))
            .setMessage(
                context.getString(
                    R.string.large_model_warning_message,
                    modelName,
                    String.format("%.0f", modelSize)
                )
            )
            .setPositiveButton(context.getString(R.string.continue_open)) { _, _ ->
                onConfirm()
            }
            .setNegativeButton(context.getString(R.string.cancel)) { _, _ ->
                onCancel?.invoke()
            }
            .setCancelable(true)
            .show()
    }
    
    /**
     * Show confirmation dialog for large models (>7GB) in Fragment
     * @param fragment Fragment to show dialog
     * @param modelName Name of the model
     * @param modelSize Size of the model in GB
     * @param onConfirm Callback when user confirms to continue
     * @param onCancel Callback when user cancels (optional)
     */
    fun show(
        fragment: Fragment,
        modelName: String,
        modelSize: Double,
        onConfirm: () -> Unit,
        onCancel: (() -> Unit)? = null
    ) {
        show(fragment.requireContext(), modelName, modelSize, onConfirm, onCancel)
    }
}
