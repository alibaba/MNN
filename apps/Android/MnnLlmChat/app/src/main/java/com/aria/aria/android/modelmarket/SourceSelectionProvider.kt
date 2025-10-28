package com.alibaba.mnnllm.android.modelmarket

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import com.alibaba.mnnllm.android.R

data class SourceProvider(val name: String, val id: String)

class SourceSelectionProvider(
    private val sourceProviders: List<SourceProvider>,
    private val onItemClicked: (SourceProvider) -> Unit
) : RecyclerView.Adapter<SourceSelectionProvider.ProviderViewHolder>() {

    private var selectedPosition = -1

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ProviderViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.list_item_provider, parent, false)
        return ProviderViewHolder(view)
    }

    override fun onBindViewHolder(holder: ProviderViewHolder, position: Int) {
        val provider = sourceProviders[position]
        holder.bind(provider, position == selectedPosition)
        holder.itemView.setOnClickListener {
            val previousSelectedPosition = selectedPosition
            selectedPosition = holder.adapterPosition
            if(previousSelectedPosition != -1) {
                notifyItemChanged(previousSelectedPosition)
            }
            notifyItemChanged(selectedPosition)
            onItemClicked(provider)
        }
    }

    override fun getItemCount(): Int = sourceProviders.size

    fun setSelected(providerId: String) {
        val index = sourceProviders.indexOfFirst { it.id == providerId }
        if (index != -1) {
            selectedPosition = index
            notifyDataSetChanged()
        }
    }

    class ProviderViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val nameTextView: TextView = itemView.findViewById(R.id.tv_provider_name)
        private val checkImageView: ImageView = itemView.findViewById(R.id.iv_check)

        fun bind(sourceProvider: SourceProvider, isSelected: Boolean) {
            nameTextView.text = sourceProvider.name
            checkImageView.visibility = if (isSelected) View.VISIBLE else View.INVISIBLE
        }
    }
} 