// Copyright (c) 2026 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.mainsettings

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import com.alibaba.mnnllm.android.R
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context

/**
 * Adapter for expandable storage list: group rows (model entry) and child rows (storage unit with delete).
 */
class StorageListAdapter(
    private val formatSize: (Long) -> String,
    private val getStatusCleanable: () -> String,
    private val getStatusTracked: () -> String,
    private val onGroupClick: (StorageListItem.Group) -> Unit,
    private val onChildDelete: (StorageListItem.Child) -> Unit
) : ListAdapter<StorageListItem, RecyclerView.ViewHolder>(StorageDiffCallback()) {

    companion object {
        private const val VIEW_TYPE_GROUP = 0
        private const val VIEW_TYPE_CHILD = 1
    }

    override fun getItemViewType(position: Int): Int = when (getItem(position)) {
        is StorageListItem.Group -> VIEW_TYPE_GROUP
        is StorageListItem.Child -> VIEW_TYPE_CHILD
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): RecyclerView.ViewHolder =
        when (viewType) {
            VIEW_TYPE_GROUP -> {
                val v = LayoutInflater.from(parent.context).inflate(R.layout.item_storage_group, parent, false)
                GroupViewHolder(v, formatSize, getStatusCleanable, getStatusTracked, onGroupClick)
            }
            VIEW_TYPE_CHILD -> {
                val v = LayoutInflater.from(parent.context).inflate(R.layout.item_storage_child, parent, false)
                ChildViewHolder(v, formatSize, onChildDelete)
            }
            else -> throw IllegalArgumentException("Unknown viewType $viewType")
        }

    override fun onBindViewHolder(holder: RecyclerView.ViewHolder, position: Int) {
        when (val item = getItem(position)) {
            is StorageListItem.Group -> (holder as GroupViewHolder).bind(item)
            is StorageListItem.Child -> (holder as ChildViewHolder).bind(item)
        }
    }

    class GroupViewHolder(
        itemView: View,
        private val formatSize: (Long) -> String,
        private val getStatusCleanable: () -> String,
        private val getStatusTracked: () -> String,
        private val onGroupClick: (StorageListItem.Group) -> Unit
    ) : RecyclerView.ViewHolder(itemView) {

        private val chevron: ImageView = itemView.findViewById(R.id.storage_group_chevron)
        private val label: TextView = itemView.findViewById(R.id.storage_group_label)
        private val size: TextView = itemView.findViewById(R.id.storage_group_size)
        private val status: TextView = itemView.findViewById(R.id.storage_group_status)

        fun bind(item: StorageListItem.Group) {
            label.text = item.entry.modelId
            size.text = formatSize(item.entry.sizeBytes)
            status.text = if (item.entry.isOrphan) getStatusCleanable() else getStatusTracked()
            chevron.rotation = if (item.expanded) 180f else 0f
            itemView.setOnClickListener { onGroupClick(item) }
        }
    }

    class ChildViewHolder(
        itemView: View,
        private val formatSize: (Long) -> String,
        private val onChildDelete: (StorageListItem.Child) -> Unit
    ) : RecyclerView.ViewHolder(itemView) {

        private val label: TextView = itemView.findViewById(R.id.storage_child_label)
        private val size: TextView = itemView.findViewById(R.id.storage_child_size)
        private val deleteBtn: View = itemView.findViewById(R.id.storage_child_delete)

        fun bind(item: StorageListItem.Child) {
            label.text = item.label
            val sizeVal = item.sizeBytes
            if (sizeVal != null && sizeVal >= 0) {
                size.visibility = View.VISIBLE
                size.text = formatSize(sizeVal)
            } else {
                size.visibility = View.GONE
            }
            deleteBtn.setOnClickListener { onChildDelete(item) }
            // Long press to copy path
            itemView.setOnLongClickListener {
                item.path?.let { path ->
                    val clipboard = itemView.context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
                    val clip = ClipData.newPlainText("Storage Path", path)
                    clipboard.setPrimaryClip(clip)
                    Toast.makeText(itemView.context, R.string.path_copied, Toast.LENGTH_SHORT).show()
                    true
                } ?: false
            }
        }
    }

    private class StorageDiffCallback : DiffUtil.ItemCallback<StorageListItem>() {
        override fun areItemsTheSame(a: StorageListItem, b: StorageListItem): Boolean {
            if (a::class != b::class) return false
            return when (a) {
                is StorageListItem.Group -> a.entry.path == (b as StorageListItem.Group).entry.path
                is StorageListItem.Child -> {
                    val cb = b as StorageListItem.Child
                    a.entry.modelId == cb.entry.modelId && a.type == cb.type && a.path == cb.path
                }
            }
        }

        override fun areContentsTheSame(a: StorageListItem, b: StorageListItem): Boolean = a == b
    }
}
