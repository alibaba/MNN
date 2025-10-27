// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.modelist

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.RecyclerView
import com.alibaba.mls.api.ModelItem
import com.alibaba.mls.api.download.DownloadInfo
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.model.Modality
import com.alibaba.mnnllm.android.model.ModelUtils
import kotlinx.coroutines.*
import java.util.Locale

class ModelListAdapter(private val items: MutableList<ModelItemWrapper>) :
    RecyclerView.Adapter<RecyclerView.ViewHolder>() {
    var initialized = false
    private var filteredItems: List<ModelItemWrapper> = items.toList()
    private var modelListListener: ModelItemListener? = null
    
    // Use HashMap for O(1) lookups instead of iterating through all holders
    private val modelItemHoldersMap: MutableMap<String, ModelItemHolder> = HashMap()
    private val modelItemHolders: MutableSet<ModelItemHolder> = HashSet()
    
    private var filterQuery: String = ""
    private var filterQueryMap = mutableMapOf("query" to "", "vendor" to "", "modality" to "", "download" to "-1")
    private var emptyView: View? = null
    
    // Coroutine scope for async filtering
    private val adapterScope = CoroutineScope(Dispatchers.Main + SupervisorJob())
    private var filterJob: Job? = null

    // Pin toggle event callback
    interface OnPinToggleListener {
        fun onPinToggle(modelId: String, isPinned: Boolean)
    }
    
    private var onPinToggleListener: OnPinToggleListener? = null
    
    fun setOnPinToggleListener(listener: OnPinToggleListener?) {
        this.onPinToggleListener = listener
    }

    fun setModelListListener(modelListListener: ModelItemListener?) {
        this.modelListListener = modelListListener
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): RecyclerView.ViewHolder {
        val view =
            LayoutInflater.from(parent.context).inflate(R.layout.recycle_item_model, parent, false)
        
        // Create a wrapped listener to handle pin events
        val wrappedListener = object : ModelItemListener {
            override fun onItemClicked(modelItem: ModelItem) {
                modelListListener?.onItemClicked(modelItem)
            }
            
            override fun onItemLongClicked(modelItem: ModelItem): Boolean {
                val modelWrapper = items.find { it.modelItem.modelId == modelItem.modelId }
                if (modelWrapper != null) {
                    val newPinState = !modelWrapper.isPinned
                    onPinToggleListener?.onPinToggle(modelItem.modelId!!, newPinState)
                    return true
                }
                return false
            }
            
            override fun onItemDeleted(modelItem: ModelItem) {
                modelListListener?.onItemDeleted(modelItem)
            }
            
            override fun onItemUpdate(modelItem: ModelItem) {
                modelListListener?.onItemUpdate(modelItem)
            }
        }
        
        val holder = ModelItemHolder(view, wrappedListener)
        modelItemHolders.add(holder)
        return holder
    }

    override fun onBindViewHolder(holder: RecyclerView.ViewHolder, position: Int) {
        val modelWrapper = getItems()[position]
        val modelItemHolder = holder as ModelItemHolder
        
        // Register holder in map for fast lookups
        modelItemHoldersMap[modelWrapper.modelItem.modelId!!] = modelItemHolder
        
        modelItemHolder.bind(modelWrapper)
    }
    
    override fun onBindViewHolder(holder: RecyclerView.ViewHolder, position: Int, payloads: MutableList<Any>) {
        if (payloads.isNotEmpty()) {
            val payload = payloads[0]
            if (payload is DownloadInfo) {
                // Handle progress update
                val modelItemHolder = holder as ModelItemHolder
                modelItemHolder.updateProgress(payload)
            } else {
                // Since all models are downloaded, we can do simpler updates
                val modelWrapper = getItems()[position]
                val modelItemHolder = holder as ModelItemHolder
                modelItemHolder.bind(modelWrapper) // Just rebind with new data
            }
        } else {
            super.onBindViewHolder(holder, position, payloads)
        }
    }
    
    override fun onViewRecycled(holder: RecyclerView.ViewHolder) {
        super.onViewRecycled(holder)
        // Remove from map when view is recycled
        val modelWrapper = holder.itemView.tag as? ModelItemWrapper
        modelWrapper?.modelItem?.modelId?.let { modelId ->
            modelItemHoldersMap.remove(modelId)
        }
    }

    override fun getItemCount(): Int {
        return getItems().size
    }

    fun updateItems(modelWrappers: List<ModelItemWrapper>) {
        items.clear()
        items.addAll(modelWrappers)
        
        // Use DiffUtil for efficient updates
        val oldFiltered = filteredItems
        filterItemsSync() // Filter synchronously for immediate update
        
        val diffCallback = ModelWrapperDiffCallback(oldFiltered, filteredItems)
        val diffResult = DiffUtil.calculateDiff(diffCallback)
        diffResult.dispatchUpdatesTo(this)
        initialized = true
        checkIfEmpty()
    }

    fun updateItem(modelId: String) {
        val currentItems = getItems()
        var position = -1
        for (i in currentItems.indices) {
            if (currentItems[i].modelItem.modelId == modelId) {
                position = i
                break
            }
        }
        if (position >= 0) {
            notifyItemChanged(position)
        }
    }

    fun updateItemHasUpdate(modelId: String, hasUpdate: Boolean) {
        // Find the item in the original items list and update it
        for (i in items.indices) {
            if (items[i].modelItem.modelId == modelId) {
                items[i].hasUpdate = hasUpdate
                break
            }
        }
        
        // Update the item in the view
        updateItem(modelId)
    }

    /**
     * Update progress for a specific model when it's being updated
     */
    fun updateProgress(modelId: String, downloadInfo: DownloadInfo) {
        val currentItems = getItems()
        var position = -1
        for (i in currentItems.indices) {
            if (currentItems[i].modelItem.modelId == modelId) {
                position = i
                break
            }
        }
        if (position >= 0) {
            notifyItemChanged(position, downloadInfo)
        }
    }

    private fun getItems(): List<ModelItemWrapper> {
        return filteredItems
    }

    private fun filterItemsSync() {
        val filtered = items.filter { modelWrapper ->
            val modelItem = modelWrapper.modelItem
            val modelNameLowerCase = modelItem.modelName!!.lowercase(Locale.getDefault())
            
            for ((key, value) in filterQueryMap) {
                if (value.isEmpty()) {
                    continue
                }
                when (key) {
                    "vendor" -> {
                        val vendor = ModelUtils.getVendor(modelNameLowerCase)
                        if (vendor != value) {
                            return@filter false
                        }
                    }
                    "modality" -> {
                        if (!Modality.checkModality(modelItem.modelId!!.lowercase(), value)) {
                            return@filter false
                        }
                    }
                    "download" -> {
                        // All models in this list are downloaded, so this filter is always true
                        // We can remove this filter or keep it for compatibility
                        if (value == "true") {
                            // Show all items since they're all downloaded
                            continue
                        }
                    }
                    else -> {
                        if (!modelNameLowerCase.contains(value.lowercase(Locale.getDefault()))) {
                            return@filter false
                        }
                    }
                }
            }
            true
        }
        filteredItems = filtered
    }

    private fun filterItems() {
        // Cancel previous filter job
        filterJob?.cancel()
        
        // Perform filtering asynchronously
        filterJob = adapterScope.launch {
            val filtered = withContext(Dispatchers.Default) {
                items.filter { modelWrapper ->
                    val modelItem = modelWrapper.modelItem
                    val modelNameLowerCase = modelItem.modelName!!.lowercase(Locale.getDefault())
                    
                    for ((key, value) in filterQueryMap) {
                        if (value.isEmpty()) {
                            continue
                        }
                        when (key) {
                            "vendor" -> {
                                val vendor = ModelUtils.getVendor(modelNameLowerCase)
                                if (vendor != value) {
                                    return@filter false
                                }
                            }
                            "modality" -> {
                                if (!Modality.checkModality(modelItem.modelId!!, value)) {
                                    return@filter false
                                }
                            }
                            "download" -> {
                                // All models are downloaded, so this filter always passes
                                if (value == "true") {
                                    continue
                                }
                            }
                            else -> {
                                if (!modelNameLowerCase.contains(value.lowercase(Locale.getDefault()))) {
                                    return@filter false
                                }
                            }
                        }
                    }
                    true
                }
            }
            
            // Back to main thread for UI updates
            if (isActive) {
                val oldFiltered = filteredItems
                filteredItems = filtered
                
                val diffCallback = ModelWrapperDiffCallback(oldFiltered, filteredItems)
                val diffResult = DiffUtil.calculateDiff(diffCallback)
                diffResult.dispatchUpdatesTo(this@ModelListAdapter)
                
                checkIfEmpty()
            }
        }
    }

    fun unfilter() {
        filterJob?.cancel()
        
        val oldFiltered = filteredItems
        filteredItems = items.toList()
        
        val diffCallback = ModelWrapperDiffCallback(oldFiltered, filteredItems)
        val diffResult = DiffUtil.calculateDiff(diffCallback)
        diffResult.dispatchUpdatesTo(this)
        
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
    }

    private fun checkIfEmpty() {
        if (!initialized) {
            return
        }
        if (emptyView != null) {
            emptyView?.visibility = if (getItemCount() == 0) View.VISIBLE else View.GONE
        }
    }
    
    // Clean up coroutines when adapter is destroyed
    fun onDestroy() {
        adapterScope.cancel()
        modelItemHoldersMap.clear()
        modelItemHolders.clear()
    }
    
    // DiffUtil callback for efficient list updates
    private class ModelWrapperDiffCallback(
        private val oldList: List<ModelItemWrapper>,
        private val newList: List<ModelItemWrapper>
    ) : DiffUtil.Callback() {
        
        override fun getOldListSize(): Int = oldList.size
        
        override fun getNewListSize(): Int = newList.size
        
        override fun areItemsTheSame(oldItemPosition: Int, newItemPosition: Int): Boolean {
            return oldList[oldItemPosition].modelItem.modelId == newList[newItemPosition].modelItem.modelId
        }
        
        override fun areContentsTheSame(oldItemPosition: Int, newItemPosition: Int): Boolean {
            val oldWrapper = oldList[oldItemPosition]
            val newWrapper = newList[newItemPosition]
            val oldItem = oldWrapper.modelItem
            val newItem = newWrapper.modelItem
            
            // Compare basic model properties including pin state
            return oldItem.modelId == newItem.modelId &&
                    oldItem.modelName == newItem.modelName &&
                    oldItem.isLocal == newItem.isLocal &&
                    oldItem.localPath == newItem.localPath &&
                    oldWrapper.downloadSize == newWrapper.downloadSize &&
                    oldWrapper.lastChatTime == newWrapper.lastChatTime &&
                    oldWrapper.downloadTime == newWrapper.downloadTime &&
                    oldWrapper.isPinned == newWrapper.isPinned // Include pin state comparison
        }
        
        override fun getChangePayload(oldItemPosition: Int, newItemPosition: Int): Any? {
            val oldWrapper = oldList[oldItemPosition]
            val newWrapper = newList[newItemPosition]
            val oldItem = oldWrapper.modelItem
            val newItem = newWrapper.modelItem
            
            if (oldItem.modelId != newItem.modelId) {
                return null // Different items, full rebind needed
            }
            
            // Create payload for specific changes to enable smooth animations
            return when {
                oldItem.modelName != newItem.modelName -> null // Full rebind needed
                oldWrapper.isPinned != newWrapper.isPinned -> "PIN_STATE_UPDATE"
                oldWrapper.lastChatTime != newWrapper.lastChatTime -> "TIME_UPDATE"
                oldWrapper.downloadSize != newWrapper.downloadSize -> "SIZE_UPDATE"
                else -> "MINOR_UPDATE"
            }
        }
    }

    companion object {
        const val TAG = "ModelListAdapter"
    }

}