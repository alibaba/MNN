package com.alibaba.mnnllm.android.utils

import android.content.Context
import android.widget.CheckBox
import com.alibaba.mnnllm.android.R
import com.google.android.material.dialog.MaterialAlertDialogBuilder

object DialogUtils {
    fun showModelDeletionDialog(
        context: Context,
        onConfirm: (keepHistory: Boolean) -> Unit
    ) {
        val keepHistory = CheckBox(context).apply {
            setText(R.string.keep_chat_history)
            isChecked = true
        }
        val padding = (24 * context.resources.displayMetrics.density).toInt()
        MaterialAlertDialogBuilder(context)
            .setTitle(R.string.confirm_delete_model_title)
            .setMessage(R.string.delete_model_history_message)
            .setView(keepHistory, padding, 0, padding, 0)
            .setPositiveButton(android.R.string.ok) { _, _ ->
                onConfirm(keepHistory.isChecked)
            }
            .setNegativeButton(android.R.string.cancel, null)
            .show()
    }

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