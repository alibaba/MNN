package com.alibaba.mnnllm.android.chat

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.recyclerview.widget.LinearLayoutManager
import com.alibaba.mnnllm.android.databinding.FragmentChooseSourceBinding
import com.alibaba.mnnllm.android.utils.BaseBottomSheetDialogFragment

class SelectSourceFragment : BaseBottomSheetDialogFragment() {
    private lateinit var onSourceSelectedListener: (String) -> Unit
    private lateinit var sourceSelectionAdapter: SourceSelectionAdapter
    private lateinit var binding: FragmentChooseSourceBinding
    private var sources: List<String> = emptyList()
    private var displayNames: List<Int> = emptyList()
    private var currentSource: String? = null

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View {
        binding = FragmentChooseSourceBinding.inflate(inflater)
        setupSourceList()
        return binding.root
    }

    private fun setupSourceList() {
        val recyclerView = binding.recyclerViewOptions
        sourceSelectionAdapter = SourceSelectionAdapter(sources, displayNames, currentSource) { selectedSource ->
            onSourceSelectedListener.invoke(selectedSource)
            dismiss()
        }
        recyclerView.apply {
            layoutManager = LinearLayoutManager(requireContext())
            adapter = sourceSelectionAdapter
        }
    }

    fun setOnSourceSelectedListener(listener: (String) -> Unit) {
        onSourceSelectedListener = listener
    }

    fun setSources(sources: List<String>, displayNames: List<Int>, current: String?) {
        this.sources = sources
        this.displayNames = displayNames
        this.currentSource = current
        if (::sourceSelectionAdapter.isInitialized) {
            sourceSelectionAdapter.updateSelectedSource(current)
        }
    }

    companion object {
        const val TAG = "SelectSourceFragment"
        
        fun newInstance(
            sources: List<String>,
            displayNames: List<Int>,
            currentSource: String? = null
        ): SelectSourceFragment {
            return SelectSourceFragment().apply {
                setSources(sources, displayNames, currentSource)
            }
        }
    }
} 