// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.privacy

import android.app.Dialog
import android.content.Context
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.CheckBox
import androidx.fragment.app.DialogFragment
import com.alibaba.mnnllm.android.R

class PrivacyPolicyDialogFragment : DialogFragment() {
    
    private var onAgreeListener: (() -> Unit)? = null
    private var onDisagreeListener: (() -> Unit)? = null
    
    companion object {
        const val TAG = "PrivacyPolicyDialog"
        
        fun newInstance(
            onAgree: (() -> Unit)? = null,
            onDisagree: (() -> Unit)? = null
        ): PrivacyPolicyDialogFragment {
            val fragment = PrivacyPolicyDialogFragment()
            fragment.onAgreeListener = onAgree
            fragment.onDisagreeListener = onDisagree
            return fragment
        }
    }
    
    override fun onCreateDialog(savedInstanceState: Bundle?): Dialog {
        val dialog = super.onCreateDialog(savedInstanceState)
        dialog.setCancelable(false)
        dialog.setCanceledOnTouchOutside(false)
        return dialog
    }
    
    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        return inflater.inflate(R.layout.dialog_privacy_policy, container, false)
    }
    
    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        
        val checkboxAgree = view.findViewById<CheckBox>(R.id.checkbox_agree)
        val btnAgree = view.findViewById<Button>(R.id.btn_agree)
        val btnDisagree = view.findViewById<Button>(R.id.btn_disagree)
        
        // Initially disable the agree button
        btnAgree.isEnabled = false
        
        // Enable/disable agree button based on checkbox state
        checkboxAgree.setOnCheckedChangeListener { _, isChecked ->
            btnAgree.isEnabled = isChecked
        }
        
        // Handle agree button click
        btnAgree.setOnClickListener {
            onAgreeListener?.invoke()
            dismiss()
        }
        
        // Handle disagree button click
        btnDisagree.setOnClickListener {
            onDisagreeListener?.invoke()
            dismiss()
        }
    }
    
    override fun onStart() {
        super.onStart()
        // Make dialog non-cancelable
        dialog?.setCancelable(false)
        dialog?.setCanceledOnTouchOutside(false)
    }
}
