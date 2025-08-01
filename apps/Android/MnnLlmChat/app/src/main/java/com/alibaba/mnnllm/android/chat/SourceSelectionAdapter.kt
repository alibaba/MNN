package com.alibaba.mnnllm.android.chat

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import com.alibaba.mnnllm.android.R

class SourceSelectionAdapter(
    private val sources: List<String>,
    private val displayNames: List<Int>,
    private var selectedSource: String?,
    private val onSourceSelected: (String) -> Unit
) : RecyclerView.Adapter<SourceSelectionAdapter.SourceViewHolder>() {

    inner class SourceViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val sourceNameTextView: TextView = itemView.findViewById(R.id.tv_source_name)
        private val checkImageView: ImageView = itemView.findViewById(R.id.iv_check)

        fun bind(source: String, displayNameRes: Int, isSelected: Boolean) {
            sourceNameTextView.text = itemView.context.getString(displayNameRes)
            checkImageView.visibility = if (isSelected) View.VISIBLE else View.INVISIBLE
            itemView.setOnClickListener {
                onSourceSelected(source)
            }
        }
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): SourceViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.list_item_source_selection, parent, false)
        return SourceViewHolder(view)
    }

    override fun onBindViewHolder(holder: SourceViewHolder, position: Int) {
        val source = sources[position]
        val displayNameRes = displayNames.getOrNull(position) ?: 0
        val isSelected = source == selectedSource
        holder.bind(source, displayNameRes, isSelected)
    }

    override fun getItemCount(): Int = sources.size

    fun updateSelectedSource(newSelectedSource: String?) {
        val previousSelectedIndex = sources.indexOf(selectedSource)
        val newSelectedIndex = sources.indexOf(newSelectedSource)
        selectedSource = newSelectedSource
        if (previousSelectedIndex != -1) {
            notifyItemChanged(previousSelectedIndex)
        }
        if (newSelectedIndex != -1) {
            notifyItemChanged(newSelectedIndex)
        }
    }
} 