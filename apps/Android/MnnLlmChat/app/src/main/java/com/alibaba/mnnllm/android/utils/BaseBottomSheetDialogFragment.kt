package com.alibaba.mnnllm.android.utils

import android.content.Context
import androidx.appcompat.app.AlertDialog
import androidx.lifecycle.lifecycleScope
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.model.ModelTypeUtils
import com.google.android.material.bottomsheet.BottomSheetDialogFragment
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

/**
 * Base BottomSheetDialogFragment with shared warning helpers.
 */
abstract class BaseBottomSheetDialogFragment : BottomSheetDialogFragment() {

    protected fun shouldWarnOpenClByExtraTags(knownExtraTags: List<String>): Boolean {
        return ModelTypeUtils.isOpenClWarningByExtraTags(knownExtraTags)
    }

    protected fun showOpenClWarningDialog(
        onProceed: () -> Unit,
        onCancel: (() -> Unit)? = null
    ) {
        showOpenClWarningDialog(requireContext(), lifecycleScope, onProceed, onCancel)
    }

//    override fun getTheme(): Int {
//        return R.style.BottomSheetDialogTheme
//    }

    companion object {
        private const val OPENCL_CONFIRM_DELAY_SECONDS = 10

        fun showOpenClWarningDialog(
            context: Context,
            scope: CoroutineScope,
            onProceed: () -> Unit,
            onCancel: (() -> Unit)? = null
        ) {
            var countdownJob: Job? = null

            val dialog = MaterialAlertDialogBuilder(context)
                .setTitle(R.string.opencl_warning_title)
                .setMessage(R.string.opencl_warning_message)
                .setNegativeButton(R.string.cancel) { _, _ ->
                    onCancel?.invoke()
                }
                .setPositiveButton(
                    context.getString(
                        R.string.opencl_warning_use_countdown,
                        OPENCL_CONFIRM_DELAY_SECONDS
                    )
                ) { _, _ ->
                    onProceed()
                }
                .create()

            dialog.setOnShowListener {
                val positiveButton = dialog.getButton(AlertDialog.BUTTON_POSITIVE)
                positiveButton.isEnabled = false
                countdownJob = scope.launch {
                    for (secondLeft in OPENCL_CONFIRM_DELAY_SECONDS downTo 1) {
                        positiveButton.text = context.getString(
                            R.string.opencl_warning_use_countdown,
                            secondLeft
                        )
                        delay(1000)
                    }
                    positiveButton.text = context.getString(R.string.opencl_warning_use)
                    positiveButton.isEnabled = true
                }
            }

            dialog.setOnDismissListener {
                countdownJob?.cancel()
                countdownJob = null
            }
            dialog.show()
        }
    }
}
