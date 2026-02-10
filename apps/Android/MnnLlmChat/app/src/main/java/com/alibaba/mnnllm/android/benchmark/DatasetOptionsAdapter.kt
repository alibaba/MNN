// Created by ruoyi.sjd on 2025/4/14.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.benchmark
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.RecyclerView
import com.alibaba.mnnllm.android.databinding.ListItemDatasetBinding

data class DatasetOptionItem(
    val id: String,
    val title: String,
    val subtitle: String,
)

class DatasetOptionsAdapter(
    private var items: List<DatasetOptionItem>
) : RecyclerView.Adapter<DatasetOptionsAdapter.OptionsViewHolder>() {

    private var listener: ((DatasetOptionItem) -> Unit)? = null

    inner class OptionsViewHolder(private val binding: ListItemDatasetBinding) :
        RecyclerView.ViewHolder(binding.root) {

        fun bind(item: DatasetOptionItem) {
            binding.tvItemTitle.text = item.title
            binding.tvItemSubtitle.text = item.subtitle
            binding.ivItemIcon.visibility = View.GONE
            this.itemView.tag = item
        }
    }

    fun setOnItemClickListener(listener: (DatasetOptionItem) -> Unit) {
        this.listener = listener
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): OptionsViewHolder {
        val binding = ListItemDatasetBinding.inflate(
            LayoutInflater.from(parent.context),
            parent,
            false
        )
        binding.root.setOnClickListener { v->
            (v.tag as? DatasetOptionItem)?.let {
                listener?.invoke(it)
            }
        }
        return OptionsViewHolder(binding)
    }

    override fun onBindViewHolder(holder: OptionsViewHolder, position: Int) {
        holder.bind(items[position])
    }


    override fun getItemCount(): Int = items.size

    fun updateData(newItems: List<DatasetOptionItem>) {
        val diffCallback = OptionDiffCallback(this.items, newItems)
        val diffResult = DiffUtil.calculateDiff(diffCallback)
        this.items = newItems
        diffResult.dispatchUpdatesTo(this)
    }

    private class OptionDiffCallback(
        private val oldList: List<DatasetOptionItem>,
        private val newList: List<DatasetOptionItem>
    ) : DiffUtil.Callback(){
        override fun getOldListSize(): Int = oldList.size
        override fun getNewListSize(): Int = newList.size

        override fun areItemsTheSame(oldItemPosition: Int, newItemPosition: Int): Boolean {
            return oldList[oldItemPosition].title == newList[newItemPosition].title
        }
        override fun areContentsTheSame(oldItemPosition: Int, newItemPosition: Int): Boolean {
            return oldList[oldItemPosition] == newList[newItemPosition] // Relies on data class equality
        }
    }
}