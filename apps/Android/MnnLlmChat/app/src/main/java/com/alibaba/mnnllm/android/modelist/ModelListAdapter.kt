// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.modelist

import android.text.TextUtils
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.recyclerview.widget.RecyclerView
import com.alibaba.mls.api.ModelItem
import com.alibaba.mls.api.download.DownloadInfo
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.model.Modality
import com.alibaba.mnnllm.android.model.ModelUtils
import java.util.Locale
import java.util.stream.Collectors

class ModelListAdapter(private val items: MutableList<ModelItem>) :
    RecyclerView.Adapter<RecyclerView.ViewHolder>() {
    private var filteredItems: List<ModelItem>? = null
    private var modelListListener: ModelItemListener? = null
    private var modelItemDownloadStatesMap: Map<String, ModelItemDownloadState>? = null
    private val modelItemHolders: MutableSet<ModelItemHolder> = HashSet()
    private var filterQuery: String = ""
    private var filterQueryMap = mutableMapOf("query" to "", "vendor" to "", "modality" to "", "download" to "-1")
    private var filterDownloaded = false
    private var emptyView: View? = null

    fun setModelListListener(modelListListener: ModelItemListener?) {
        this.modelListListener = modelListListener
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): RecyclerView.ViewHolder {
        val view =
            LayoutInflater.from(parent.context).inflate(R.layout.recycle_item_model, parent, false)
        val holder = ModelItemHolder(view, modelListListener!!)
        modelItemHolders.add(holder)
        return holder
    }

    override fun onBindViewHolder(holder: RecyclerView.ViewHolder, position: Int) {
        (holder as ModelItemHolder).bind(
            getItems()[position],
            modelItemDownloadStatesMap!![getItems()[position].modelId]
        )
    }

    override fun getItemCount(): Int {
        return getItems().size
    }

    fun updateItems(
        hfModelItems: List<ModelItem>,
        modelItemDownloadStatesMap: Map<String, ModelItemDownloadState>?
    ) {
        this.modelItemDownloadStatesMap = modelItemDownloadStatesMap
        items.clear()
        items.addAll(hfModelItems)
        filterItems()
        checkIfEmpty()
    }

    fun updateItem(modelId: String) {
        val currentItems = getItems()
        var position = -1
        for (i in currentItems.indices) {
            if (currentItems[i].modelId == modelId) {
                position = i
                break
            }
        }
        if (position >= 0) {
            notifyItemChanged(position)
        }
    }

    fun updateProgress(modelId: String?, downloadInfo: DownloadInfo) {
        for (modelItemHolder in modelItemHolders) {
            if (modelItemHolder.itemView.tag == null) {
                continue
            }
            val tempModelId = (modelItemHolder.itemView.tag as ModelItem).modelId
            if (TextUtils.equals(tempModelId, modelId)) {
                modelItemHolder.updateProgress(downloadInfo)
            }
        }
    }

    private fun getItems(): List<ModelItem> {
        return if (filteredItems != null) filteredItems!! else items
    }

    private fun filterItems() {
        val filtered = items.stream()
            .filter { hfModelItem: ModelItem ->
                val modelItemState = modelItemDownloadStatesMap!![hfModelItem.modelId]
                val modelNameLowerCase = hfModelItem.modelName!!.lowercase(Locale.getDefault())
                for ((key, value) in filterQueryMap) {
                    if (value.isEmpty()) {
                        continue
                    }
                    if (key == "vendor") {
                        val vendor = ModelUtils.getVendor(modelNameLowerCase)
                        if (vendor != value) {
                            return@filter false
                        }
                    } else if (key == "modality") {
                        if (!Modality.checkModality(modelNameLowerCase, value)) {
                            return@filter false
                        }
                    }  else if (key == "download") {
                        if (value == "true" && modelItemState?.downloadInfo?.downlodaState != DownloadInfo.DownloadSate.COMPLETED
                            && !hfModelItem.isLocal) {
                            return@filter false
                        }
                    } else if (!modelNameLowerCase.contains(value.lowercase(Locale.getDefault()))) {
                        return@filter false
                    }
                }
                return@filter true
            }
            .collect(Collectors.toList())
        if (filtered.size != items.size) {
            this.filteredItems = filtered
        } else {
            this.filteredItems = null
        }
        notifyDataSetChanged()
        checkIfEmpty()
    }

    fun unfilter() {
        this.filteredItems = null
        notifyDataSetChanged()
        checkIfEmpty()
    }

    fun setFilter(filterQuery: String) {
        this.filterQuery = filterQuery
        this.filterQueryMap["query"] = filterQuery
        filterItems()
    }

    fun filterVendor(vendorFilter: String) {
        this.filterQueryMap["vendor"] = vendorFilter
        filterItems()
    }

    fun filterModality(modality: String) {
        this.filterQueryMap["modality"] = modality
        filterItems()
    }

    fun filterDownloadState(downloadState: String) {
        this.filterQueryMap["download"] = downloadState
        filterItems()
    }

    fun setEmptyView(emptyView: View) {
        this.emptyView = emptyView
        checkIfEmpty()
    }

    private fun checkIfEmpty() {
        if (emptyView != null) {
            emptyView?.visibility = if (getItemCount() == 0) View.VISIBLE else View.GONE
        }
    }
}