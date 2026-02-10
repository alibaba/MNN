package com.alibaba.mnnllm.android.modelmarket

import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.TextView
import androidx.constraintlayout.widget.ConstraintLayout
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.lifecycle.ViewModelProvider
import com.alibaba.mls.api.download.DownloadState
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.databinding.FragmentModelMarketBinding
import com.alibaba.mnnllm.android.main.MainActivity
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.alibaba.mnnllm.android.main.FilterComponent
import com.alibaba.mnnllm.android.mainsettings.MainSettings
import com.alibaba.mnnllm.android.model.Modality
import com.alibaba.mnnllm.android.model.ModelVendors
import com.alibaba.mnnllm.android.model.ModelTypeUtils
import com.alibaba.mnnllm.android.modelsettings.DropDownMenuHelper
import com.alibaba.mnnllm.android.utils.Searchable
import com.alibaba.mnnllm.android.widgets.ModelSwitcherView
import android.widget.Toast
import com.alibaba.mnnllm.android.utils.LargeModelConfirmationDialog

class ModelMarketFragment : Fragment(), ModelMarketItemListener, Searchable {

    private var _binding: FragmentModelMarketBinding? = null
    private val binding get() = _binding!!

    private lateinit var viewModel: ModelMarketViewModel
    private lateinit var adapter: ModelMarketAdapter
    private lateinit var recyclerView: RecyclerView
    private lateinit var emptyView: View
    private lateinit var loadingView: View
    private lateinit var errorView: View
    private lateinit var errorText: TextView
    private var modalityFilterIndex = 0
    private var vendorFilterIndex = 0
    private lateinit var sourceSwitcher: ModelSwitcherView

    private lateinit var rootLayout: ConstraintLayout
    private var filterComponent: FilterComponent? = null
    private var filterContainerView: View? = null
    private var currentFilterState = FilterState()

    // Track whether quick filter tags have been initialized
    private var quickFilterTagsInitialized = false
    
    // Save current search query state
    private var currentSearchQuery: String = ""

    // Implement Searchable interface
    override fun onSearchQuery(query: String) {
        currentSearchQuery = query
        currentFilterState = currentFilterState.copy(searchQuery = query)
        viewModel.applyFilters(currentFilterState)
    }

    override fun onSearchCleared() {
        currentSearchQuery = ""
        currentFilterState = currentFilterState.copy(searchQuery = "")
        viewModel.applyFilters(currentFilterState)
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentModelMarketBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        viewModel = ViewModelProvider(this).get(ModelMarketViewModel::class.java)

        setupCustomToolbar()
        setupRecyclerView()
        setupEmptyView()
        setupLoadingView()
        setupErrorView()

        // Observe loading state
        viewModel.isLoading.observe(viewLifecycleOwner) { isLoading ->
            if (isLoading) {
                onLoading()
            }
        }

        // Observe error state
        viewModel.loadError.observe(viewLifecycleOwner) { errorMessage ->
            if (errorMessage != null) {
                onListLoadError(errorMessage)
            }
        }

        // Observe filtered models
        viewModel.models.observe(viewLifecycleOwner) { models ->
            adapter.submitList(models)
            onListAvailable(models)
            updateEmptyViewVisibility(models)
            
            // Initialize quick filter tags when data is first loaded
            if (!quickFilterTagsInitialized && models.isNotEmpty()) {
                setupQuickFilterTagsAfterDataLoad()
                quickFilterTagsInitialized = true
            }
        }

        // Observe progress updates
        viewModel.progressUpdate.observe(viewLifecycleOwner) { (modelId, downloadInfo) ->
            adapter.updateProgress(modelId, downloadInfo)
        }

        // Observe item updates
        viewModel.itemUpdate.observe(viewLifecycleOwner) { modelId ->
            adapter.updateItem(modelId)
        }

        rootLayout = view.findViewById(R.id.model_market_root_layout)
        recyclerView.layoutManager = LinearLayoutManager(context)
    }

    private fun setupCustomToolbar() {
        val appBarContent = requireActivity().findViewById<ViewGroup>(R.id.app_bar_content)
        
        // Setup Filter Components only (no more TabLayout)
        if (filterContainerView == null) {
            filterContainerView = layoutInflater.inflate(R.layout.layout_toolbar_filters_content, appBarContent, false)
            setupFilterComponent()
        }
        if (filterContainerView?.parent == null) {
            appBarContent?.addView(filterContainerView)
        }
    }
    
    private fun setupFilterComponent() {
        filterContainerView?.let { container ->
            // Create FilterComponent using the main activity context and container
            if (filterComponent == null) {
                // We need to create a temporary MainActivity instance for FilterComponent
                // Since FilterComponent expects MainActivity, we'll create our own filter logic here
                setupFilterButtons(container)
            }
        }
    }
    
    private fun setupFilterButtons(container: View) {
        val filterDownloadState = container.findViewById<TextView>(R.id.filter_download_state)
        val filterModality = container.findViewById<TextView>(R.id.filter_modality)
        val filterVendor = container.findViewById<TextView>(R.id.filter_vendor)
        sourceSwitcher = container.findViewById<ModelSwitcherView>(R.id.source_switcher)
        val filterButtonLayout = container.findViewById<ViewGroup>(R.id.filter_button_layout)
        
        // Setup download state filter
        filterDownloadState.setOnClickListener {
            filterDownloadState.isSelected = !filterDownloadState.isSelected
            val downloadState = if (filterDownloadState.isSelected) DownloadState.COMPLETED else null
            updateFilterDownloadState(downloadState)
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
        
        // Setup filter dialog button
        filterButtonLayout.setOnClickListener {
            showFilterDialog()
        }
        
        // Initialize filter button states
        updateFilterButtonStates()
    }

    
    private fun updateFilterVendor(vendor: String?) {
        val vendors = if (vendor != null) listOf(vendor) else emptyList()
        currentFilterState = FilterState(
            tagKeys = currentFilterState.tagKeys,
            vendors = vendors,
            size = currentFilterState.size,
            modality = currentFilterState.modality,
            downloadState = currentFilterState.downloadState,
            source = currentFilterState.source,
            searchQuery = currentFilterState.searchQuery
        )
        viewModel.applyFilters(currentFilterState)
        updateFilterButtonStates()
        updateQuickFilterButtonStates()
    }
    
    private fun updateFilterModality(modality: String?) {
        currentFilterState = FilterState(
            tagKeys = currentFilterState.tagKeys,
            vendors = currentFilterState.vendors,
            size = currentFilterState.size,
            modality = modality,
            downloadState = currentFilterState.downloadState,
            source = currentFilterState.source,
            searchQuery = currentFilterState.searchQuery
        )
        viewModel.applyFilters(currentFilterState)
        updateFilterButtonStates()
        updateQuickFilterButtonStates()
    }
    
    private fun updateFilterDownloadState(downloadState: Int?) {
        currentFilterState = FilterState(
            tagKeys = currentFilterState.tagKeys,
            vendors = currentFilterState.vendors,
            size = currentFilterState.size,
            modality = currentFilterState.modality,
            downloadState = downloadState,
            source = currentFilterState.source,
            searchQuery = currentFilterState.searchQuery
        )
        viewModel.applyFilters(currentFilterState)
        updateFilterButtonStates()
        updateQuickFilterButtonStates()
    }
    
    private fun updateFilterSource(source: String?) {
        MainSettings.setDownloadProvider(requireContext(), source!!)
        Log.d(TAG, "selectedSource: $source got from preference ${MainSettings.getDownloadProviderString(requireContext())}")
        viewModel.loadModels()
    }

    private fun setupRecyclerView() {
        recyclerView = binding.modelMarketRecyclerView
        adapter = ModelMarketAdapter(this)
        recyclerView.layoutManager = LinearLayoutManager(context)
        recyclerView.adapter = adapter
    }

    private fun setupEmptyView() {
        emptyView = binding.modelMarketEmptyView
        val clearFiltersButton = emptyView.findViewById<View>(R.id.btn_clear_filters)
        clearFiltersButton.setOnClickListener {
            clearAllFilters()
        }
    }

    private fun setupLoadingView() {
        loadingView = binding.modelMarketLoadingView
    }

    private fun setupErrorView() {
        errorView = binding.modelMarketFailedView
        errorText = errorView.findViewById(R.id.tv_error_text)
        errorView.setOnClickListener {
            // Retry loading when user clicks error view
            viewModel.loadModels()
        }
    }

    private fun onLoading() {
        // Don't show loading if we already have data
        if (adapter.itemCount > 0) {
            return
        }
        errorView.visibility = View.GONE
        emptyView.visibility = View.GONE
        recyclerView.visibility = View.GONE
        loadingView.visibility = View.VISIBLE
    }

    private fun onListAvailable(models: List<ModelMarketItemWrapper>) {
        errorView.visibility = View.GONE
        loadingView.visibility = View.GONE
        
        // RecyclerView visibility will be managed by updateEmptyViewVisibility
    }

    private fun onListLoadError(error: String) {
        // Don't show error if we already have data
        if (adapter.itemCount > 0) {
            return
        }
        
        errorText.text = getString(R.string.loading_failed_click_tor_retry, error)
        errorView.visibility = View.VISIBLE
        loadingView.visibility = View.GONE
        recyclerView.visibility = View.GONE
        emptyView.visibility = View.GONE
    }

    private fun updateEmptyViewVisibility(models: List<ModelMarketItemWrapper>) {
        val hasFilters = hasActiveFilters()
        
        // Show empty view only when there are active filters but no results
        if (models.isEmpty() && hasFilters) {
            emptyView.visibility = View.VISIBLE
            recyclerView.visibility = View.GONE
        } else {
            emptyView.visibility = View.GONE
            // Make sure recyclerView is visible when we have data or no filters
            recyclerView.visibility = View.VISIBLE
        }
    }

    private fun hasActiveFilters(): Boolean {
        return currentFilterState.tagKeys.isNotEmpty() ||
               currentFilterState.vendors.isNotEmpty() ||
               currentFilterState.size != null ||
               currentFilterState.modality != null ||
               currentFilterState.downloadState != null ||
               currentFilterState.searchQuery.isNotEmpty()
    }

    private fun clearAllFilters() {
        // Clear all filter states (including search query)
        currentFilterState = FilterState()
        
        // Clear search query variable
        currentSearchQuery = ""
        
        // Reset filter UI elements
        resetFilterUIElements()
        
        // Apply empty filters to refresh the model list
        viewModel.applyFilters(currentFilterState)
        
        // Clear search view in MainActivity
        (requireActivity() as? MainActivity)?.clearSearch()
        
        // Update UI
        updateFilterButtonStates()
        updateQuickFilterButtonStates()
    }

    private fun resetFilterUIElements() {
        filterContainerView?.let { container ->
            // Reset dropdown filter indices
            modalityFilterIndex = 0
            vendorFilterIndex = 0
            
            // Reset UI text and selection states
            val filterModality = container.findViewById<TextView>(R.id.filter_modality)
            val filterVendor = container.findViewById<TextView>(R.id.filter_vendor)
            val filterDownloadState = container.findViewById<TextView>(R.id.filter_download_state)
            
            filterModality.text = getString(R.string.modality_menu_title)
            filterModality.isSelected = false
            
            filterVendor.text = getString(R.string.vendor_menu_title)
            filterVendor.isSelected = false
            
            filterDownloadState.isSelected = false
        }
    }

    private fun removeCustomToolbar() {
        val appBarContent = requireActivity().findViewById<ViewGroup>(R.id.app_bar_content)
        if (filterContainerView != null) {
            appBarContent?.removeView(filterContainerView)
        }
    }


    override fun onHiddenChanged(hidden: Boolean) {
        super.onHiddenChanged(hidden)
        Log.d(TAG, "onHiddenChanged: hidden = $hidden")
        if (hidden) {
            Log.d(TAG, "onHiddenChanged: removing custom toolbar")
            removeCustomToolbar()
        } else {
            Log.d(TAG, "onHiddenChanged: setting up custom toolbar")
            setupCustomToolbar()
            // Restore search state if there was an active search
            restoreSearchStateIfNeeded()
        }
    }
    
    override fun onResume() {
        super.onResume()
        Log.d(TAG, "onResume: isHidden = $isHidden")
        // Also restore search state on resume (for initial load)
        onHiddenChanged(isHidden)
        restoreSearchStateIfNeeded()
    }
    
    override fun onPause() {
        super.onPause()
        Log.d(TAG, "onPause: isHidden = $isHidden")
    }
    
    override fun setUserVisibleHint(isVisibleToUser: Boolean) {
        super.setUserVisibleHint(isVisibleToUser)
        Log.d(TAG, "setUserVisibleHint: isVisibleToUser = $isVisibleToUser")
        if (!isVisibleToUser && isAdded) {
            Log.d(TAG, "setUserVisibleHint: removing custom toolbar")
            removeCustomToolbar()
        }
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

    private fun showFilterDialog() {
        val dialog = FilterDialogFragment()
        val availableVendors = viewModel.getAvailableVendors()
        val availableTags = viewModel.getAvailableTags()
        dialog.setAvailableVendors(availableVendors)
        dialog.setAvailableTags(availableTags)
        dialog.setCurrentFilterState(currentFilterState)
        dialog.setOnConfirmListener { filterState ->
            currentFilterState = FilterState(
                tagKeys = filterState.tagKeys,
                vendors = filterState.vendors,
                size = filterState.size,
                modality = currentFilterState.modality,
                downloadState = filterState.downloadState,
                source = currentFilterState.source,
                searchQuery = currentFilterState.searchQuery
            )
            viewModel.applyFilters(currentFilterState)
            updateFilterButtonStates()
            updateQuickFilterButtonStates()
        }
        dialog.show(childFragmentManager, "FilterDialog")
    }
    
    private fun setupQuickFilterTags(container: View) {
        val quickFilterLayout = container.findViewById<LinearLayout>(R.id.quick_filter_tags_layout)
        quickFilterLayout.removeAllViews()
        viewModel.getQuickFilterTags().forEach { tag ->
            val quickFilterButton = TextView(requireContext()).apply {
                text = tag.getDisplayText()
                setPadding(16, 0, 16, 0)
                setBackgroundResource(R.drawable.bg_filter_chip)
                setTextColor(ContextCompat.getColorStateList(requireContext(), R.color.filter_button_tint))
                textSize = 12f
                gravity = android.view.Gravity.CENTER_VERTICAL
                isClickable = true
                isFocusable = true

                val layoutParams = LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    resources.getDimensionPixelSize(R.dimen.filter_chip_height)
                ).apply {
                    marginEnd = resources.getDimensionPixelSize(R.dimen.filter_chip_margin_end)
                }
                this.layoutParams = layoutParams

                setOnClickListener {
                    toggleQuickFilter(tag)
                }
            }
            quickFilterLayout.addView(quickFilterButton)
        }
    }
    
    private fun toggleQuickFilter(tag: Tag) {
        val currentTags = currentFilterState.tagKeys.toMutableList()
        if (currentTags.contains(tag.key)) {
            // Deselect if already selected
            currentTags.clear()
        } else {
            // Only select this tag
            currentTags.clear()
            currentTags.add(tag.key)
        }
        currentFilterState = FilterState(
            tagKeys = currentTags,
            vendors = currentFilterState.vendors,
            size = currentFilterState.size,
            modality = currentFilterState.modality,
            downloadState = currentFilterState.downloadState,
            source = currentFilterState.source,
            searchQuery = currentFilterState.searchQuery
        )
        viewModel.applyFilters(currentFilterState)
        updateFilterButtonStates()
        updateQuickFilterButtonStates()
    }
    
    private fun updateQuickFilterButtonStates() {
        filterContainerView?.let { container ->
            val quickFilterLayout = container.findViewById<LinearLayout>(R.id.quick_filter_tags_layout)
            val quickFilterTags = viewModel.getQuickFilterTags()
            val selectedKey = currentFilterState.tagKeys.firstOrNull()
            for (i in 0 until quickFilterLayout.childCount) {
                val button = quickFilterLayout.getChildAt(i) as? TextView
                val tag = quickFilterTags.getOrNull(i)
                if (button != null && tag != null) {
                    button.isSelected = (tag.key == selectedKey)
                }
            }
        }
    }

    private fun updateFilterButtonStates() {
        filterContainerView?.let { container ->
            val filterDownloadState = container.findViewById<TextView>(R.id.filter_download_state)
            val filterModality = container.findViewById<TextView>(R.id.filter_modality)
            val filterVendor = container.findViewById<TextView>(R.id.filter_vendor)
            val filterButtonLayout = container.findViewById<ViewGroup>(R.id.filter_button_layout)
            
            // Update download state selection
            filterDownloadState.isSelected = currentFilterState.downloadState != null
            filterDownloadState.text = when (currentFilterState.downloadState) {
                DownloadState.DOWNLOAD_SUCCESS -> getString(R.string.download_state_completed)
                DownloadState.NOT_START -> getString(R.string.download_state_not_start)
                DownloadState.DOWNLOADING -> getString(R.string.download_state_downloading)
                DownloadState.DOWNLOAD_PAUSED -> getString(R.string.download_state_paused)
                DownloadState.DOWNLOAD_FAILED -> getString(R.string.download_state_failed)
                DownloadState.DOWNLOAD_CANCELLED -> getString(R.string.download_state_cancelled)
                DownloadState.PREPARING -> getString(R.string.download_state_preparing)
                else -> getString(R.string.download_state)
            }
            
            // Update modality selection and text
            filterModality.isSelected = currentFilterState.modality != null
            filterModality.text = currentFilterState.modality ?: getString(R.string.modality_menu_title)
            
            // Update vendor selection and text
            filterVendor.isSelected = currentFilterState.vendors.isNotEmpty()
            filterVendor.text = when {
                currentFilterState.vendors.isEmpty() -> getString(R.string.vendor_menu_title)
                currentFilterState.vendors.size == 1 -> currentFilterState.vendors.first()
                else -> "${currentFilterState.vendors.size} vendors"
            }
            
            // Update source switcher text
            if (::sourceSwitcher.isInitialized) {
                sourceSwitcher.text = MainSettings.getDownloadProviderString(requireContext())
            }
            
            // Update filter button selection based on whether any advanced filters are active
            val hasAdvancedFilters = currentFilterState.tagKeys.isNotEmpty()
                    || currentFilterState.size != null
                    || currentFilterState.vendors.isNotEmpty()
            filterButtonLayout.isSelected = hasAdvancedFilters
        }
    }

    override fun onActionClicked(item: ModelMarketItemWrapper) {
        //add log
        Log.d(TAG, "onActionClicked: " + item.modelMarketItem.modelId)
        val downloadInfo = item.downloadInfo
        when (downloadInfo.downloadState) {
            DownloadState.COMPLETED -> {
                // Check if model has update first
                if (downloadInfo.hasUpdate) {
                    // Start update download
                    viewModel.updateModel(item.modelMarketItem)
                    return
                }
                
                // Check if it's a voice model (TTS or ASR)
                if (ModelTypeUtils.isTtsModelByTags(item.modelMarketItem.tags) || ModelTypeUtils.isAsrModelByTags(item.modelMarketItem.tags)) {
                    // For voice models, clicking the item sets it as default
                    handleVoiceModelClick(item.modelMarketItem)
                } else {
                    // For other models, check if it's a large model before running
                    if (item.modelMarketItem.sizeB > 10.0) {
                        // Show confirmation dialog for large models
                        LargeModelConfirmationDialog.show(
                            fragment = this,
                            modelName = item.modelMarketItem.modelName,
                            modelSize = item.modelMarketItem.sizeB,
                            onConfirm = {
                                // User confirmed, proceed with running the model
                                (requireActivity() as MainActivity).runModel(null, item.modelMarketItem.modelId, null)
                            },
                            onCancel = {
                                // User cancelled, do nothing
                            }
                        )
                    } else {
                        // Model is not large, run directly
                        (requireActivity() as MainActivity).runModel(null, item.modelMarketItem.modelId, null)
                    }
                }
            }
            DownloadState.NOT_START,
            DownloadState.FAILED -> {
                viewModel.startDownload(item.modelMarketItem)
            }
            DownloadState.PAUSED -> {
                viewModel.startDownload(item.modelMarketItem) // resume
            }
            DownloadState.DOWNLOADING -> {
                viewModel.pauseDownload(item.modelMarketItem)
            }
        }
    }

    override fun onDownloadOrResumeClicked(item: ModelMarketItemWrapper) {
        viewModel.startDownload(item.modelMarketItem)
    }

    override fun onPauseClicked(item: ModelMarketItemWrapper) {
        viewModel.pauseDownload(item.modelMarketItem)
    }

    override fun onDeleteClicked(item: ModelMarketItemWrapper) {
        viewModel.deleteModel(item.modelMarketItem)
    }

    override fun onUpdateClicked(item: ModelMarketItemWrapper) {
        viewModel.updateModel(item.modelMarketItem)
    }

    override fun onDefaultVoiceModelChanged(item: ModelMarketItemWrapper) {
        handleVoiceModelClick(item.modelMarketItem)
    }

    private fun handleVoiceModelClick(modelMarketItem: ModelMarketItem) {
        if (ModelTypeUtils.isTtsModelByTags(modelMarketItem.tags)) {
            handleTtsModelClick(modelMarketItem)
        } else if (ModelTypeUtils.isAsrModelByTags(modelMarketItem.tags)) {
            handleAsrModelClick(modelMarketItem)
        }
    }

    private fun handleTtsModelClick(modelMarketItem: ModelMarketItem) {
        val context = requireContext()
        val modelId = modelMarketItem.modelId
        
        // Check if already default TTS model
        val isCurrentDefault = MainSettings.isDefaultTtsModel(context, modelId)

        if (isCurrentDefault) {
            // Already default model, show toast
            Toast.makeText(context, getString(R.string.tts_model_set_as_default), Toast.LENGTH_SHORT).show()
        } else {
            // Set as default TTS model
            MainSettings.setDefaultTtsModel(context, modelId)
            Toast.makeText(context, getString(R.string.tts_model_set_as_default), Toast.LENGTH_SHORT).show()

            // Reload models to reflect the change in default status
            // Refresh adapter to update all checkbox states, ensuring only one is selected
            refreshAdapterForVoiceModelChange()
        }
    }

    private fun handleAsrModelClick(modelMarketItem: ModelMarketItem) {
        val context = requireContext()
        val modelId = modelMarketItem.modelId

        // Check if already default ASR model
        val isCurrentDefault = MainSettings.isDefaultAsrModel(context, modelId)

        if (isCurrentDefault) {
            // Already default model, show toast
            Toast.makeText(context, getString(R.string.already_default_asr_model), Toast.LENGTH_SHORT).show()
        } else {
            // Set as default ASR model
            MainSettings.setDefaultAsrModel(context, modelId)
            Toast.makeText(context, getString(R.string.default_asr_model_set, modelMarketItem.modelName), Toast.LENGTH_SHORT).show()

            // Reload models to reflect the change in default status
            // Refresh adapter to update all voice model checkbox states, ensuring only one is selected
            refreshAdapterForVoiceModelChange()
        }
    }

    /**
     * Refresh adapter to update all checkbox states, ensuring only one is selected
     */
    private fun refreshAdapterForVoiceModelChange() {
        //Notify adapter to refresh all items so each item rechecks if it's the default model
        adapter.notifyDataSetChanged()
    }

    override fun onDestroyView() {
        super.onDestroyView()
        removeCustomToolbar()
        recyclerView.adapter = null
        _binding = null
        filterComponent = null
        filterContainerView = null
    }

    private fun setupQuickFilterTagsAfterDataLoad() {
        filterContainerView?.let { container ->
            setupQuickFilterTags(container)
        }
    }

    /**
     * When source occurs change, used to update data
     */
    fun onSourceChanged() {
        viewModel.loadModels()
    }

    companion object {
        private var lastClickTime: Long = -1
        private const val TAG = "ModelMarketFragment"
    }
}