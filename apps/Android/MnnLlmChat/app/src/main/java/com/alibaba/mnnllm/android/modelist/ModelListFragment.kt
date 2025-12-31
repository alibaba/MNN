// Created by ruoyi.sjd on 2025/1/13.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.modelist

import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import android.widget.Toast
import androidx.fragment.app.Fragment
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.alibaba.mnnllm.android.main.MainActivity
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.chat.ChatRouter
import com.alibaba.mnnllm.android.utils.PreferenceUtils.isFilterDownloaded
import com.alibaba.mnnllm.android.utils.PreferenceUtils
import com.alibaba.mnnllm.android.model.Modality
import com.alibaba.mnnllm.android.model.ModelVendors
import com.alibaba.mnnllm.android.modelsettings.DropDownMenuHelper
import com.alibaba.mnnllm.android.utils.Searchable
import com.alibaba.mnnllm.android.utils.LargeModelConfirmationDialog
import com.alibaba.mnnllm.android.modelist.ModelListManager

class ModelListFragment : Fragment(), ModelListContract.View, Searchable {
    
    companion object {
        private const val TAG = "ModelListFragment"
    }
    private lateinit var modelListRecyclerView: RecyclerView
    private lateinit var modelListLoadingView: View
    private lateinit var modelListErrorView: View
    private lateinit var modelListEmptyView: View
    private var toolbarFiltersContent: View? = null

    override var adapter: ModelListAdapter? = null
        private set

    private var modelListPresenter: ModelListPresenter? = null
    private val modelItemList: MutableList<ModelItemWrapper> = mutableListOf()

    private var modelListErrorText: TextView? = null
    private var loadingMessageText: TextView? = null

    private var filterDownloaded = false
    private var filterQuery = ""
    
    // Save current search query state  
    private var currentSearchQuery: String = ""
    
    // Filter indices for dropdown selections
    private var modalityFilterIndex = 0
    private var vendorFilterIndex = 0

    // Implement Searchable interface
    override fun onSearchQuery(query: String) {
        currentSearchQuery = query
        filterQuery = query
        adapter?.setFilter(query)
    }

    override fun onSearchCleared() {
        currentSearchQuery = ""
        adapter?.unfilter()
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        val view = inflater.inflate(R.layout.fragment_modellist, container, false)
        modelListRecyclerView = view.findViewById(R.id.model_list_recycler_view)
        modelListLoadingView = view.findViewById(R.id.model_list_loading_view)
        modelListErrorView = view.findViewById(R.id.model_list_failed_view)
        modelListEmptyView = view.findViewById(R.id.model_list_empty_view)
        modelListErrorText = modelListErrorView.findViewById(R.id.tv_error_text)
        loadingMessageText = modelListLoadingView.findViewById(R.id.tv_loading_message)
        modelListRecyclerView.setLayoutManager(
            LinearLayoutManager(
                context,
                LinearLayoutManager.VERTICAL,
                false
            )
        )
        adapter = ModelListAdapter(modelItemList)
        adapter!!.initialized = false
        adapter!!.setEmptyView(modelListEmptyView)

        modelListRecyclerView.setAdapter(adapter)
        modelListPresenter = ModelListPresenter(requireContext(), this)
        adapter!!.setModelListListener(modelListPresenter)
        
        // Set pin event listener
        adapter!!.setOnPinToggleListener(object : ModelListAdapter.OnPinToggleListener {
            override fun onPinToggle(modelId: String, isPinned: Boolean) {
                handlePinToggle(modelId, isPinned)
            }
        })
        
        filterDownloaded = isFilterDownloaded(context)
        adapter!!.setFilter(filterQuery)
        adapter!!.filterDownloadState(filterDownloaded.toString())

        // Show loading view initially to prevent flash of empty list
        modelListLoadingView.visibility = View.VISIBLE
        modelListRecyclerView.visibility = View.GONE
        modelListErrorView.visibility = View.GONE
        modelListEmptyView.visibility = View.GONE
        
        modelListPresenter!!.onCreate()
        return view
    }

    /**
     * Handle pin toggle action
     */
    private fun handlePinToggle(modelId: String, isPinned: Boolean) {
        try {
            // Check if we're at the top of the list before making changes
            val layoutManager = modelListRecyclerView.layoutManager as? LinearLayoutManager
            val shouldScrollToTop = layoutManager?.let {
                val firstVisiblePosition = it.findFirstVisibleItemPosition()
                val firstCompletelyVisiblePosition = it.findFirstCompletelyVisibleItemPosition()
                // Consider "at top" if first item is visible and we're unpinning (item will move down)
                (firstVisiblePosition <= 2 || firstCompletelyVisiblePosition <= 1) && !isPinned
            } ?: false

            if (isPinned) {
                PreferenceUtils.pinModel(requireContext(), modelId)
                Toast.makeText(requireContext(), getString(R.string.model_pinned), Toast.LENGTH_SHORT).show()
            } else {
                PreferenceUtils.unpinModel(requireContext(), modelId)
                Toast.makeText(requireContext(), getString(R.string.model_unpinned), Toast.LENGTH_SHORT).show()
            }

            // Notify presenter to refresh the list with new pin state
            modelListPresenter?.handlePinStateChange(isPinned)

            // If we were at the top and unpinned an item, scroll back to top after update
            if (shouldScrollToTop) {
                modelListRecyclerView.post {
                    modelListRecyclerView.scrollToPosition(0)
                }
            }

        } catch (e: Exception) {
            Log.e(TAG, "Failed to toggle pin state for model: $modelId", e)
            Toast.makeText(requireContext(), getString(R.string.pin_toggle_failed), Toast.LENGTH_SHORT).show()
        }
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        setupCustomToolbar()
    }

    private fun setupCustomToolbar() {
        val appBarContent = requireActivity().findViewById<ViewGroup>(R.id.app_bar_content)
//        if (toolbarFiltersContent == null) {
//            toolbarFiltersContent = layoutInflater.inflate(R.layout.layout_toolbar_filters_content, appBarContent, false)
//            setupFilterButtons()
//        }
//        if (toolbarFiltersContent?.parent == null) {
//            appBarContent?.addView(toolbarFiltersContent)
//        }
    }
    
    private fun setupFilterButtons() {
        toolbarFiltersContent?.let { container ->
            val filterDownloadState = container.findViewById<TextView>(R.id.filter_download_state)
            val filterModality = container.findViewById<TextView>(R.id.filter_modality)
            val filterVendor = container.findViewById<TextView>(R.id.filter_vendor)
            
            // Setup download state filter - initialize with current state
            filterDownloadState.isSelected = filterDownloaded
            filterDownloadState.setOnClickListener {
                filterDownloaded = !filterDownloaded
                filterDownloadState.isSelected = filterDownloaded
                updateFilterDownloadState()
            }
            
            // Setup modality filter
            filterModality.setOnClickListener {
                val modalityList = mutableListOf<String>().apply {
                    add(getString(R.string.all))
                    addAll(Modality.modalitySelectorList)
                }
                DropDownMenuHelper.showDropDownMenu(
                    context = requireContext(),
                    anchorView = filterModality,
                    items = modalityList,
                    currentIndex = modalityFilterIndex,
                    onItemSelected = { index, item ->
                        val hasSelected = index != 0
                        modalityFilterIndex = index
                        val modality = if (index == 0) null else item.toString()
                        filterModality.text = if (modality == null) getString(R.string.modality_menu_title) else item.toString()
                        filterModality.isSelected = hasSelected
                        updateFilterModality(modality)
                    }
                )
            }
            
            // Setup vendor filter
            filterVendor.setOnClickListener {
                val vendorList = mutableListOf<String>().apply {
                    add(getString(R.string.all))
                    addAll(ModelVendors.vendorList)
                }
                DropDownMenuHelper.showDropDownMenu(
                    context = requireContext(),
                    anchorView = filterVendor,
                    items = vendorList,
                    currentIndex = vendorFilterIndex,
                    onItemSelected = { index, item ->
                        val hasSelected = index != 0
                        vendorFilterIndex = index
                        val vendor = if (index == 0) null else item.toString()
                        filterVendor.text = if (vendor == null) getString(R.string.vendor_menu_title) else item.toString()
                        filterVendor.isSelected = hasSelected
                        updateFilterVendor(vendor)
                    }
                )
            }
            
            // Source filter removed - not needed for local model list
        }
    }
    
    private fun updateFilterDownloadState() {
        adapter?.filterDownloadState(filterDownloaded.toString())
    }
    
    private fun updateFilterModality(modality: String?) {
        adapter?.filterModality(modality ?: "")
    }
    
    private fun updateFilterVendor(vendor: String?) {
        adapter?.filterVendor(vendor ?: "")
    }
    
    // updateFilterSource method removed - not needed for local model list

    private fun removeCustomToolbar() {
        if (toolbarFiltersContent != null) {
            val appBarContent = requireActivity().findViewById<ViewGroup>(R.id.app_bar_content)
            appBarContent?.removeView(toolbarFiltersContent)
        }
    }

    override fun onHiddenChanged(hidden: Boolean) {
        super.onHiddenChanged(hidden)
        if (hidden) {
            removeCustomToolbar()
        } else {
            setupCustomToolbar()
            restoreSearchStateIfNeeded()
        }
    }
    
    override fun onResume() {
        super.onResume()

        restoreSearchStateIfNeeded()
    }
    
    /**
     * Restore search state if there was an active search query
     */
    fun restoreSearchStateIfNeeded() {
        if (currentSearchQuery.isNotEmpty()) {
            // Post with delay to ensure menu and SearchView are ready
            view?.postDelayed({
                val mainActivity = requireActivity() as? MainActivity
                mainActivity?.setSearchQuery(currentSearchQuery)
            }, 100)
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        modelListPresenter!!.onDestroy()
        adapter?.onDestroy() // Clean up adapter resources
        removeCustomToolbar()
                toolbarFiltersContent = null
    }

    override fun onListAvailable() {
        modelListErrorView.visibility = View.GONE
        modelListLoadingView.visibility = View.GONE
        
        // Only show recycler view if adapter has items
        if (adapter!!.itemCount > 0) {
            modelListRecyclerView.visibility = View.VISIBLE
            modelListEmptyView.visibility = View.GONE
        } else {
            modelListRecyclerView.visibility = View.GONE
            modelListEmptyView.visibility = View.VISIBLE
        }
    }

    override fun onLoading() {
        if (adapter!!.itemCount > 0) {
            return
        }
        modelListErrorView.visibility = View.GONE
        modelListLoadingView.visibility = View.VISIBLE
        modelListRecyclerView.visibility = View.GONE
    }

    override fun onListLoadError(error: String?) {
        if (adapter!!.itemCount > 0) {
            return
        }
        modelListErrorText!!.text = getString(R.string.loading_failed_click_tor_retry, error)
        modelListErrorView.visibility = View.VISIBLE
        modelListLoadingView.visibility = View.GONE
        modelListRecyclerView.visibility = View.GONE
    }

    override fun onBuiltinModelsCopyProgress(current: Int, total: Int, message: String) {
        requireActivity().runOnUiThread {
            loadingMessageText?.text = message
        }
    }

    override fun runModel(destPath:String?, modelId: String?) {
        // Check if model is larger than 7GB before running
        val modelItem = ModelListManager.getModelIdModelMap()[modelId]
        val modelMarketItem = (modelItem?.modelMarketItem as? com.alibaba.mnnllm.android.modelmarket.ModelMarketItem)

        if (modelMarketItem != null && modelMarketItem.sizeB > 10.0) {
            // Show confirmation dialog for large models
            LargeModelConfirmationDialog.show(
                fragment = this,
                modelName = modelMarketItem.modelName,
                modelSize = modelMarketItem.sizeB,
                onConfirm = {
                    // User confirmed, proceed with running the model
                    ChatRouter.startRun(requireContext(), modelId!!, destPath, null)
                },
                onCancel = {
                    // User cancelled, do nothing
                }
            )
        } else {
            // Model is not large or size info not available, run directly
            ChatRouter.startRun(requireContext(), modelId!!, destPath, null)
        }
    }
}