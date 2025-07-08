package com.alibaba.mnnllm.android.chat

import android.view.View
import android.widget.ImageView
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import com.alibaba.mls.api.ModelItem
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.utils.ModelListManager
import com.alibaba.mnnllm.android.widgets.ModelAvatarView
import com.alibaba.mnnllm.android.widgets.TagsLayout

class ModelSelectionViewHolder(
    itemView: View,
    private val onModelClicked: (ModelListManager.ModelItemWrapper) -> Unit
) : RecyclerView.ViewHolder(itemView) {

    private val modelAvatar: ModelAvatarView = itemView.findViewById(R.id.model_avatar)
    private val modelNameTextView: TextView = itemView.findViewById(R.id.tv_model_name)
    private val tagsLayout: TagsLayout = itemView.findViewById(R.id.tags_layout)
    private val checkImageView: ImageView = itemView.findViewById(R.id.iv_check)

    private var currentModelWrapper: ModelListManager.ModelItemWrapper? = null

    init {
        itemView.setOnClickListener {
            currentModelWrapper?.let { onModelClicked(it) }
        }
    }

    fun bind(modelWrapper: ModelListManager.ModelItemWrapper, isSelected: Boolean) {
        currentModelWrapper = modelWrapper
        val modelItem = modelWrapper.modelItem
        
        // Set model name
        modelNameTextView.text = modelItem.modelName ?: modelItem.modelId
        
        // Set model avatar
        modelAvatar.setModelName(modelItem.modelName ?: modelItem.modelId ?: "")
        
        // Set tags (simplified version from ModelItemHolder)
        val tags = getDisplayTags(modelItem)
        tagsLayout.setTags(tags)
        
        // Set selection state
        checkImageView.visibility = if (isSelected) View.VISIBLE else View.INVISIBLE
    }

    private fun getDisplayTags(modelItem: ModelItem): List<String> {
        val tags = mutableListOf<String>()
        
        // Add source tag
        val source = getModelSource(modelItem.modelId)
        if (source != null) {
            tags.add(source)
        }
        
        // Use getTags() which prioritizes market tags from model_market.json
        val marketTags = modelItem.getTags()
        
        // Add local/downloaded status or market tags
        if (modelItem.isLocal) {
            tags.add(itemView.context.getString(R.string.local))
        } else if (marketTags.isNotEmpty()) {
            // If we have market tags, use them directly (they're already user-friendly)
            tags.addAll(marketTags.take(2)) // Limit to 2 market tags
        }
        
        // Limit total tags to 3 for better UI layout
        return tags.take(3)
    }

    /**
     * Extract source information from modelId
     */
    private fun getModelSource(modelId: String?): String? {
        return when {
            modelId == null -> null
            modelId.startsWith("HuggingFace/") || modelId.contains("taobao-mnn") -> itemView.context.getString(R.string.huggingface)
            modelId.startsWith("ModelScope/") -> itemView.context.getString(R.string.modelscope)
            modelId.startsWith("Modelers/") -> itemView.context.getString(R.string.modelers)
            else -> null
        }
    }
} 