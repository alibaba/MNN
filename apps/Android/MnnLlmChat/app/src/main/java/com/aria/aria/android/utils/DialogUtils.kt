package com.alibaba.mnnllm.android.utils

import android.content.Context
import com.alibaba.mnnllm.android.R
import com.google.android.material.dialog.MaterialAlertDialogBuilder

object DialogUtils {
    fun showDeleteConfirmationDialog(
        context: Context,
        onConfirm: () -> Unit
    ) {
        MaterialAlertDialogBuilder(context)
            .setTitle(R.string.confirm_delete_model_title)
            .setMessage(R.string.confirm_delete_model_message)
            .setPositiveButton(android.R.string.ok) { _, _ ->
                onConfirm()
            }
            .setNegativeButton(android.R.string.cancel, null)
            .show()
    }
} 