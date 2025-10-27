package com.alibaba.mnnllm.android.modelmarket

import android.util.Log
import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import com.alibaba.mls.api.download.DownloadInfo
import com.alibaba.mnnllm.android.R
import java.util.Locale

class ModelMarketAdapter(
    private val fragmentListener: ModelMarketItemListener
) : ListAdapter<ModelMarketItemWrapper, MarketItemHolder>(ModelDiffCallback()) {

    private var originalList: List<ModelMarketItemWrapper> = emptyList()
    private var searchQuery: String = ""
    private var voiceModelChangedCallback: ((MarketHolderVoiceDelegate.VoiceModelType, String) -> Unit)? = null

    // Use the market item listener directly
    private val holderListener = fragmentListener

    /**
     * Set callback to be executed when a voice model is set as default
     */
    fun setVoiceModelChangedCallback(callback: (MarketHolderVoiceDelegate.VoiceModelType, String) -> Unit) {
        voiceModelChangedCallback = callback
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): MarketItemHolder {
        val view = LayoutInflater.from(parent.context).inflate(R.layout.recycle_item_market, parent, false)
        // Create the holder and set the voice model callback
        val holder = MarketItemHolder(view, holderListener)
        voiceModelChangedCallback?.let { callback ->
            holder.setVoiceModelChangedCallback(callback)
        }
        return holder
    }

    override fun onBindViewHolder(holder: MarketItemHolder, position: Int) {
        val wrapper = getItem(position)
        holder.bind(wrapper)
    }

    override fun onBindViewHolder(holder: MarketItemHolder, position: Int, payloads: MutableList<Any>) {
        if (payloads.isEmpty()) {
            super.onBindViewHolder(holder, position, payloads)
        } else {
            val downloadInfo = payloads[0] as DownloadInfo
            holder.updateProgress(downloadInfo)
        }
    }

    override fun submitList(list: List<ModelMarketItemWrapper>?) {
        originalList = list ?: emptyList()
        super.submitList(applySearchFilter(originalList))
    }

    fun setSearchQuery(query: String) {
        searchQuery = query
        super.submitList(applySearchFilter(originalList))
    }

    fun clearSearch() {
        searchQuery = ""
        super.submitList(originalList)
    }

    private fun applySearchFilter(list: List<ModelMarketItemWrapper>): List<ModelMarketItemWrapper> {
        if (searchQuery.isEmpty()) {
            return list
        }
        
        return list.filter { wrapper ->
            val modelMarketItem = wrapper.modelMarketItem
            val searchLower = searchQuery.lowercase(Locale.getDefault())
            
            // Search in model name, description, tags, etc.
            modelMarketItem.modelName.lowercase(Locale.getDefault()).contains(searchLower) ||
            modelMarketItem.description?.lowercase(Locale.getDefault())?.contains(searchLower) == true ||
            modelMarketItem.tags.any { it.lowercase(Locale.getDefault()).contains(searchLower) }
        }
    }

    fun updateProgress(modelId: String, downloadInfo: DownloadInfo) {
        val position = currentList.indexOfFirst { it.modelMarketItem.modelId == modelId }
        if (position != -1) {
            getItem(position).downloadInfo = downloadInfo
            notifyItemChanged(position, downloadInfo)
        }
    }

    fun updateItem(modelId: String) {
        val position = currentList.indexOfFirst { it.modelMarketItem.modelId == modelId }
        if (position != -1) {
            notifyItemChanged(position)
        }
    }



    companion object {
        const val TAG = "ModelMarketAdapter"
    }

    class ModelDiffCallback : DiffUtil.ItemCallback<ModelMarketItemWrapper>() {
        override fun areItemsTheSame(oldItem: ModelMarketItemWrapper, newItem: ModelMarketItemWrapper): Boolean {
            return oldItem.modelMarketItem.modelId == newItem.modelMarketItem.modelId
        }

        override fun areContentsTheSame(oldItem: ModelMarketItemWrapper, newItem: ModelMarketItemWrapper): Boolean {
            // Compare relevant fields for UI changes
            return oldItem == newItem
        }
    }
} 