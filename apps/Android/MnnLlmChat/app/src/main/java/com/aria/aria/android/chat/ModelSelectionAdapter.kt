package com.alibaba.mnnllm.android.chat

import android.util.Log
import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.RecyclerView
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.modelist.ModelItemWrapper
import com.alibaba.mnnllm.android.modelist.ModelListManager

class ModelSelectionAdapter(
    private var modelWrappers: List<ModelItemWrapper> = emptyList()
) : RecyclerView.Adapter<ModelSelectionViewHolder>() {

    private var onModelSelectedListener: ((ModelItemWrapper) -> Unit)? = null
    private var selectedPosition = -1
    private var selectedModelId: String? = null

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ModelSelectionViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.list_item_model_selection, parent, false)
        return ModelSelectionViewHolder(view) { modelWrapper ->
            handleModelSelection(modelWrapper)
        }
    }

    override fun onBindViewHolder(holder: ModelSelectionViewHolder, position: Int) {
        val wrapper = modelWrappers[position]
        val isSelected = position == selectedPosition || 
                         wrapper.modelItem.modelId == selectedModelId
        Log.d(TAG, "onBindViewHolder wrapperId: ${wrapper.modelItem.modelId}, selectedModelId: $selectedModelId")
        holder.bind(wrapper, isSelected)
    }

    override fun getItemCount(): Int = modelWrappers.size

    private fun handleModelSelection(modelWrapper: ModelItemWrapper) {
        val position = modelWrappers.indexOfFirst { 
            it.modelItem.modelId == modelWrapper.modelItem.modelId 
        }
        
        if (position != -1) {
            val previousSelectedPosition = selectedPosition
            selectedPosition = position
            selectedModelId = modelWrapper.modelItem.modelId
            
            // Update UI
            if (previousSelectedPosition != -1) {
                notifyItemChanged(previousSelectedPosition)
            }
            notifyItemChanged(selectedPosition)
            
            // Notify listener
            onModelSelectedListener?.invoke(modelWrapper)
        }
    }

    fun updateData(newModelWrappers: List<ModelItemWrapper>) {
        val diffCallback = ModelWrapperDiffCallback(this.modelWrappers, newModelWrappers)
        val diffResult = DiffUtil.calculateDiff(diffCallback)
        this.modelWrappers = newModelWrappers
        
        // Update selected position if the selected model still exists
        if (selectedModelId != null) {
            selectedPosition = newModelWrappers.indexOfFirst { 
                it.modelItem.modelId == selectedModelId 
            }
        }
        
        diffResult.dispatchUpdatesTo(this)
    }

    fun setOnModelSelectedListener(listener: (ModelItemWrapper) -> Unit) {
        this.onModelSelectedListener = listener
    }

    fun setSelectedModel(modelId: String?) {
        selectedModelId = modelId
        selectedPosition = if (modelId != null) {
            modelWrappers.indexOfFirst { it.modelItem.modelId == modelId }
        } else {
            -1
        }
        notifyDataSetChanged()
    }

    companion object {
        const val TAG = "ModelSelectionAdapter"
    }

    private class ModelWrapperDiffCallback(
        private val oldList: List<ModelItemWrapper>,
        private val newList: List<ModelItemWrapper>
    ) : DiffUtil.Callback() {
        
        override fun getOldListSize(): Int = oldList.size
        
        override fun getNewListSize(): Int = newList.size
        
        override fun areItemsTheSame(oldItemPosition: Int, newItemPosition: Int): Boolean {
            return oldList[oldItemPosition].modelItem.modelId == newList[newItemPosition].modelItem.modelId
        }
        
        override fun areContentsTheSame(oldItemPosition: Int, newItemPosition: Int): Boolean {
            return oldList[oldItemPosition] == newList[newItemPosition]
        }
    }
} 