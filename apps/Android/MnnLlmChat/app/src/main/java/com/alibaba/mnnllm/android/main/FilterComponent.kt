// Created by ruoyi.sjd on 2025/5/22.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.main

import android.widget.TextView
import com.alibaba.mls.api.download.DownloadInfo
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.model.Modality
import com.alibaba.mnnllm.android.modelsettings.DropDownMenuHelper
import com.alibaba.mnnllm.android.model.ModelVendors
import com.alibaba.mnnllm.android.utils.PreferenceUtils

class FilterComponent(private val activity: MainActivity) {
    private var vendorFilterListener: ((String?) -> Unit)? = null
    private var downloadStateFilterListener: ((String) -> Unit)? = null
    private var modalityFilterListener: ((String?) -> Unit)? = null

    private val textFilterDownloadState:TextView
    private val textFilterModality:TextView
    private val textFilterVendor:TextView
    private var vendorIndex = 0
    private var modalityIndex = 0

    init {
        textFilterDownloadState = activity.findViewById(R.id.filter_download_state)
        textFilterDownloadState.isSelected = PreferenceUtils.isFilterDownloaded(activity)
        textFilterDownloadState.setOnClickListener{
            onFilterDownloadStateClick()
        }

        textFilterModality = activity.findViewById(R.id.filter_modality)
        textFilterModality.setOnClickListener {
            onFilterModalityClick()
        }

        textFilterVendor = activity.findViewById(R.id.filter_vendor)
        textFilterVendor.setOnClickListener {
            onFilterVendorClick()
        }
    }

    private fun onFilterVendorClick() {
        DropDownMenuHelper.showDropDownMenu(activity,
            textFilterVendor,
            ModelVendors.vendorList.toMutableList().apply {
                add(0, activity.getString(R.string.all))
            },
            currentIndex = vendorIndex,
            onItemSelected = { index, item ->
                val hasSelected = index != 0
                vendorIndex = index
                textFilterVendor.text = if(vendorIndex == 0) activity.getString(R.string.vendor_menu_title) else  item.toString()
                vendorFilterListener?.invoke(if (vendorIndex == 0) null else item.toString())
                textFilterVendor.isSelected = hasSelected
            }
        )
    }

    private fun onFilterModalityClick() {
        DropDownMenuHelper.showDropDownMenu(activity,
            textFilterModality,
            Modality.modalitySelectorList.toMutableList().apply {
                add(0, activity.getString(R.string.all))
            },
            currentIndex = modalityIndex,
            onItemSelected = { index, item ->
                val hasSelected = index != 0
                modalityIndex = index
                textFilterModality.text = if(!hasSelected) activity.getString(R.string.modality_menu_title) else  item.toString()
                modalityFilterListener?.invoke(if (!hasSelected) null else item.toString())
                textFilterModality.isSelected = hasSelected
            }
        )
    }

    private fun onFilterDownloadStateClick() {
        textFilterDownloadState.isSelected = !textFilterDownloadState.isSelected
        PreferenceUtils.setFilterDownloaded(activity, textFilterDownloadState.isSelected)
        downloadStateFilterListener?.invoke(if (textFilterDownloadState.isSelected) "true" else "false")
    }

    fun addVendorFilterListener(listener: (String?) -> Unit) {
        this.vendorFilterListener = listener
    }

    fun addModalityFilterListener(listener: (String?) -> Unit) {
        this.modalityFilterListener = listener
    }

    fun addDownloadFilterListener(listener: (String) -> Unit) {
        this.downloadStateFilterListener = listener
    }
}