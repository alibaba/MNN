// Created by ruoyi.sjd on 2025/5/13.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.chat

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.recyclerview.widget.LinearLayoutManager
import com.alibaba.mnnllm.android.databinding.FragmentChooseModelBinding
import com.alibaba.mnnllm.android.modelist.ModelItemWrapper
import com.alibaba.mnnllm.android.modelist.ModelListManager
import com.alibaba.mnnllm.android.utils.BaseBottomSheetDialogFragment

class SelectModelFragment : BaseBottomSheetDialogFragment() {
    private lateinit var onModelSelectedListener: (ModelItemWrapper) -> Unit
    private lateinit var modelSelectionAdapter: ModelSelectionAdapter
    private lateinit var binding: FragmentChooseModelBinding
    private var modelWrappers: List<ModelItemWrapper> = emptyList()
    private var modelFilter: ((ModelItemWrapper) -> Boolean)? = null
    private var currentModelId: String? = null

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View {
        binding = FragmentChooseModelBinding.inflate(inflater)
        setupModelList()
        return binding.root
    }

    private fun setupModelList() {
        val recyclerView = binding.recyclerViewOptions
        modelSelectionAdapter = ModelSelectionAdapter(emptyList())
        modelSelectionAdapter.setOnModelSelectedListener { modelWrapper ->
            onModelSelectedListener.invoke(modelWrapper)
            dismiss()
        }
        recyclerView.apply {
            layoutManager = LinearLayoutManager(requireContext())
            adapter = modelSelectionAdapter
        }
        loadModelsData()
    }

    fun setOnModelSelectedListener(listener: (ModelItemWrapper) -> Unit) {
        onModelSelectedListener = listener
    }

    fun setModelWrappers(modelWrappers: List<ModelItemWrapper>) {
        this.modelWrappers = modelWrappers
        if (::modelSelectionAdapter.isInitialized) {
            loadModelsData()
        }
    }

    fun setModelFilter(filter: ((ModelItemWrapper) -> Boolean)?) {
        this.modelFilter = filter
        if (::modelSelectionAdapter.isInitialized) {
            loadModelsData()
        }
    }

    fun setCurrentModelId(modelId: String?) {
        this.currentModelId = modelId
        if (::modelSelectionAdapter.isInitialized) {
            modelSelectionAdapter.setSelectedModel(modelId)
        }
    }

    private fun loadModelsData() {
        val filteredModels = if (modelFilter != null) {
            modelWrappers.filter { modelFilter!!.invoke(it) }
        } else {
            modelWrappers
        }
        modelSelectionAdapter.updateData(filteredModels)
        
        // Set the current selected model after updating data
        currentModelId?.let { 
            modelSelectionAdapter.setSelectedModel(it)
        }
    }

    companion object {
        const val TAG = "SelectModelFragment"
        
        fun newInstance(
            modelWrappers: List<ModelItemWrapper>,
            filter: ((ModelItemWrapper) -> Boolean)? = null,
            currentModelId: String? = null
        ): SelectModelFragment {
            return SelectModelFragment().apply {
                setModelWrappers(modelWrappers)
                setModelFilter(filter)
                setCurrentModelId(currentModelId)
            }
        }
    }
}