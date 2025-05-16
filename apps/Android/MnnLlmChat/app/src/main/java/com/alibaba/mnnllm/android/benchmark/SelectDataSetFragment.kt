// Created by ruoyi.sjd on 2025/5/13.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.benchmark

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.recyclerview.widget.LinearLayoutManager
import com.alibaba.mnnllm.android.databinding.FragmentChooseDatasetBinding
import com.google.android.material.bottomsheet.BottomSheetDialogFragment

class SelectDataSetFragment:BottomSheetDialogFragment() {
    private lateinit var onItemClickListener: (OptionItem) -> Unit
    private lateinit var optionsAdapter: DatasetOptionsAdapter
    private lateinit var binding: FragmentChooseDatasetBinding
    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View? {
        binding = FragmentChooseDatasetBinding.inflate(inflater)
        setupLoadDataSet()
        return binding.root
    }

    private fun setupLoadDataSet() {
        val recyclerView = binding.recyclerViewOptions
        optionsAdapter = DatasetOptionsAdapter(emptyList())
        optionsAdapter.setOnItemClickListener(onItemClickListener)
        recyclerView.apply {
            layoutManager = LinearLayoutManager(requireContext())
            adapter = optionsAdapter
        }
        loadOptionsData()
    }

    fun setOnItemClickListener(listener: (OptionItem) -> Unit) {
        onItemClickListener = listener
    }

    private fun loadOptionsData() {
        val sampleOptions = listOf(
            OptionItem(id = "multi_modal", title = "多模态数据集", subtitle = "包含语音、文本、视频数据共 30条"),
            OptionItem(id = "mmlu", title = "MMLU 文本数据集", subtitle = "包含文本数据共 100条"),
            OptionItem(id = "needle_bench", title = " NeeldeBench长文本测试数据集", subtitle = " 用于测试长文本能力"),
        )
        optionsAdapter.updateData(sampleOptions)
    }

    companion object {
        const val TAG = "SelectDataSetFragment" // Tag for FragmentManager
    }
}