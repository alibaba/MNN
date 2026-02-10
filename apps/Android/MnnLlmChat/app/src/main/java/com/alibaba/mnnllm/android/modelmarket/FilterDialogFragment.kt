package com.alibaba.mnnllm.android.modelmarket

import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import com.alibaba.mls.api.download.DownloadState
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.utils.BaseBottomSheetDialogFragment
import com.google.android.material.chip.Chip
import com.google.android.material.chip.ChipGroup

class FilterDialogFragment : BaseBottomSheetDialogFragment() {

    private var listener: ((FilterState) -> Unit)? = null
    private var availableVendors: List<String> = emptyList()
    private var availableTags: List<Tag> = emptyList()
    private var currentFilterState: FilterState? = null

    fun setOnConfirmListener(listener: (FilterState) -> Unit) {
        this.listener = listener
    }

    fun setAvailableVendors(vendors: List<String>) {
        this.availableVendors = vendors
    }

    fun setAvailableTags(tags: List<Tag>) {
        this.availableTags = tags
    }

    fun setCurrentFilterState(filterState: FilterState) {
        this.currentFilterState = filterState
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        return inflater.inflate(R.layout.dialog_fragment_filter, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        val tagsChipGroup = view.findViewById<ChipGroup>(R.id.tags_chip_group)
        val sizeChipGroup = view.findViewById<ChipGroup>(R.id.size_chip_group)
        val vendorChipGroup = view.findViewById<ChipGroup>(R.id.vendor_chip_group)
        val downloadStateChipGroup = view.findViewById<ChipGroup>(R.id.download_state_chip_group)
        val confirmButton = view.findViewById<Button>(R.id.confirm_button)
        val clearButton = view.findViewById<Button>(R.id.clear_button)
        val doneButton = view.findViewById<Button>(R.id.settings_done)

        // Setup chip groups
        setupTagChipGroup(tagsChipGroup, availableTags, true)
        setupStringChipGroup(sizeChipGroup, listOf("<1B", "1B-5B", "5B-15B", ">15B"), true)
        setupStringChipGroup(vendorChipGroup, availableVendors, true)
        setupDownloadStateChipGroup(downloadStateChipGroup)

        // Setup real-time filtering listeners first (after chips are created)
        setupRealTimeFiltering(tagsChipGroup, sizeChipGroup, vendorChipGroup, downloadStateChipGroup)

        // Restore previous filter state (this will trigger listeners and update the list)
        currentFilterState?.let { filterState ->
            restoreTagSelections(tagsChipGroup, filterState.tagKeys)
            restoreStringSelections(sizeChipGroup, listOfNotNull(filterState.size))
            restoreStringSelections(vendorChipGroup, filterState.vendors)
            restoreDownloadStateSelection(downloadStateChipGroup, filterState.downloadState)
        }

        confirmButton.setOnClickListener {
            val selectedTagKey = getSelectedTagKey(tagsChipGroup)
            val selectedSize = getSelectedChipsText(sizeChipGroup).firstOrNull()
            val selectedVendors = getSelectedChipsText(vendorChipGroup)
            val downloadState = getDownloadStateValue(downloadStateChipGroup)
            listener?.invoke(FilterState(selectedTagKey?.let { listOf(it) } ?: emptyList(), selectedVendors, selectedSize, downloadState = downloadState))
            dismiss()
        }
        doneButton.setOnClickListener {
            dismiss()
        }
        clearButton.setOnClickListener {
            tagsChipGroup.clearCheck()
            sizeChipGroup.clearCheck()
            vendorChipGroup.clearCheck()
            downloadStateChipGroup.clearCheck()
            listener?.invoke(FilterState(emptyList(), emptyList(), null, downloadState = null))
        }
    }

    private fun setupTagChipGroup(chipGroup: ChipGroup, tags: List<Tag>, singleSelection: Boolean) {
        chipGroup.isSingleSelection = true // always single selection for tags
        tags.forEach { tag ->
            val chip = LayoutInflater.from(requireContext())
                .inflate(R.layout.chip_filter_item, chipGroup, false) as Chip
            chip.text = tag.getDisplayText()
            chip.tag = tag.key // Store the key in the chip's tag for retrieval
            chipGroup.addView(chip)
        }
    }

    private fun setupStringChipGroup(chipGroup: ChipGroup, items: List<String>, singleSelection: Boolean) {
        chipGroup.isSingleSelection = singleSelection
        items.forEach { item ->
            val chip = LayoutInflater.from(requireContext())
                .inflate(R.layout.chip_filter_item, chipGroup, false) as Chip
            chip.text = item
            chipGroup.addView(chip)
        }
    }

    private fun setupDownloadStateChipGroup(chipGroup: ChipGroup) {
        chipGroup.isSingleSelection = true
        val downloadStates = listOf(
            DownloadState.DOWNLOAD_SUCCESS to getString(R.string.download_state_completed),
            DownloadState.NOT_START to getString(R.string.download_state_not_start),
            DownloadState.DOWNLOADING to getString(R.string.download_state_downloading)
        )
        
        downloadStates.forEach { (stateValue, stateText) ->
            val chip = LayoutInflater.from(requireContext())
                .inflate(R.layout.chip_filter_item, chipGroup, false) as Chip
            chip.text = stateText
            chip.tag = stateValue
            chipGroup.addView(chip)
        }
    }

    private fun getSelectedChipsText(chipGroup: ChipGroup): List<String> {
        return chipGroup.checkedChipIds.map { chipGroup.findViewById<Chip>(it).text.toString() }
    }

    private fun getSelectedTagKey(chipGroup: ChipGroup): String? {
        return chipGroup.checkedChipIds.firstOrNull()?.let { chipGroup.findViewById<Chip>(it).tag as? String }
    }

    private fun restoreTagSelections(chipGroup: ChipGroup, selectedTagKeys: List<String>) {
        val selectedKey = selectedTagKeys.firstOrNull()
        for (i in 0 until chipGroup.childCount) {
            val chip = chipGroup.getChildAt(i) as? Chip
            chip?.let {
                val tagKey = it.tag as? String
                it.isChecked = (tagKey != null && tagKey == selectedKey)
            }
        }
    }

    private fun restoreStringSelections(chipGroup: ChipGroup, selectedItems: List<String>) {
        for (i in 0 until chipGroup.childCount) {
            val chip = chipGroup.getChildAt(i) as? Chip
            chip?.let {
                if (selectedItems.contains(it.text.toString())) {
                    it.isChecked = true
                }
            }
        }
    }

    private fun setupRealTimeFiltering(
        tagsChipGroup: ChipGroup,
        sizeChipGroup: ChipGroup,
        vendorChipGroup: ChipGroup,
        downloadStateChipGroup: ChipGroup
    ) {
        val updateFilter: () -> Unit = {
            val selectedTagKey = getSelectedTagKey(tagsChipGroup)
            val selectedSize = getSelectedChipsText(sizeChipGroup).firstOrNull()
            val selectedVendors = getSelectedChipsText(vendorChipGroup)
            val downloadState = getDownloadStateValue(downloadStateChipGroup)
            Log.d("FilterDialogFragment", "Selected tag key: $selectedTagKey, Selected size: $selectedSize, Selected vendors: $selectedVendors, Download state: $downloadState")
            listener?.invoke(FilterState(selectedTagKey?.let { listOf(it) } ?: emptyList(), selectedVendors, selectedSize, downloadState = downloadState))
        }

        tagsChipGroup.setOnCheckedStateChangeListener { _, _ ->
            updateFilter()
        }
        sizeChipGroup.setOnCheckedStateChangeListener { _, _ ->
            updateFilter()
        }
        vendorChipGroup.setOnCheckedStateChangeListener { _, _ ->
            updateFilter()
        }
        downloadStateChipGroup.setOnCheckedStateChangeListener { _, _ ->
            updateFilter()
        }

        addChipListeners(tagsChipGroup, updateFilter)
        addChipListeners(sizeChipGroup, updateFilter)
        addChipListeners(vendorChipGroup, updateFilter)
        addChipListeners(downloadStateChipGroup, updateFilter)
    }

    private fun addChipListeners(chipGroup: ChipGroup, updateFilter: () -> Unit) {
        for (i in 0 until chipGroup.childCount) {
            val chip = chipGroup.getChildAt(i) as? Chip
            chip?.setOnCheckedChangeListener { _, _ ->
                updateFilter()
            }
        }
    }

    private fun restoreDownloadStateSelection(chipGroup: ChipGroup, downloadState: Int?) {
        if (downloadState == null) return
        for (i in 0 until chipGroup.childCount) {
            val chip = chipGroup.getChildAt(i) as? Chip
            chip?.let {
                val chipStateValue = it.tag as? Int
                it.isChecked = (chipStateValue == downloadState)
            }
        }
    }

    private fun getDownloadStateValue(chipGroup: ChipGroup): Int? {
        val checkedChipId = chipGroup.checkedChipId
        if (checkedChipId == -1) return null
        val chip = chipGroup.findViewById<Chip>(checkedChipId)
        return chip?.tag as? Int
    }
}