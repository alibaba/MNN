package com.alibaba.mnnllm.android.modelist

import android.util.Log
import android.view.MenuItem
import android.view.View
import android.view.View.OnLongClickListener
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.PopupMenu
import androidx.recyclerview.widget.RecyclerView
import com.alibaba.mls.api.ModelItem
import com.alibaba.mls.api.download.ModelDownloadManager
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.model.ModelUtils
import com.alibaba.mnnllm.android.modelsettings.SettingsBottomSheetFragment
import com.alibaba.mnnllm.android.utils.ModelListManager
import com.alibaba.mnnllm.android.utils.FileUtils
import com.alibaba.mnnllm.android.widgets.ModelAvatarView
import com.alibaba.mnnllm.android.widgets.TagsLayout
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import kotlinx.coroutines.MainScope
import kotlinx.coroutines.launch
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.io.File

class ModelItemHolder(
    itemView: View, 
    private val modelItemListener: ModelItemListener,
    private val enableLongClick: Boolean = true
) : RecyclerView.ViewHolder(itemView), View.OnClickListener, OnLongClickListener {
    private var tvModelName: TextView
    private var tvModelTitle: TextView
    private var tvModelSubtitle: TextView
    private var tvStatus: TextView
    private var tvTimeInfo: TextView

    private val headerSection: ModelAvatarView
    private val tagsLayout: TagsLayout
    private val pinnedOverlay: View // Pinned overlay

    private var currentModelWrapper: ModelListManager.ModelItemWrapper? = null
    private val modelDownloadManager = ModelDownloadManager.getInstance(itemView.context)

    init {
        itemView.setOnClickListener(this)
        if (enableLongClick) {
            itemView.setOnLongClickListener(this)
        }
        tvModelName = itemView.findViewById(R.id.tvModelName)
        tvModelTitle = itemView.findViewById(R.id.tvModelTitle)
        tvModelSubtitle = itemView.findViewById(R.id.tvModelSubtitle)
        tvStatus = itemView.findViewById(R.id.tvStatus)
        tvTimeInfo = itemView.findViewById(R.id.tvTimeInfo)
        headerSection = itemView.findViewById(R.id.header_section_title)
        tagsLayout = itemView.findViewById(R.id.tagsLayout)
        pinnedOverlay = itemView.findViewById(R.id.pinned_overlay)
    }

    private fun displayTimeInfo(modelWrapper: ModelListManager.ModelItemWrapper) {
        val lastChatTime = modelWrapper.lastChatTime
        
        // 1. If there hasn't been any chat, do not display
        if (lastChatTime <= 0) {
            tvTimeInfo.visibility = View.GONE
            return
        }
        
        // 2. Check if the chat happened on the same day
        val now = System.currentTimeMillis()
        val chatDate = Date(lastChatTime)
        val today = Date(now)
        
        // Determine whether it's the same day
        val isSameDay = isSameDay(chatDate, today)
        
        val formattedTime = if (isSameDay) {
            // 2.1 Chat occurred today, display hours and minutes, e.g., 8:30
            val timeFormat = SimpleDateFormat("H:mm", Locale.getDefault())
            timeFormat.format(chatDate)
        } else {
            // 2.2 Chat did not occur today, display date, supports both Chinese and English
            val locale = Locale.getDefault()
            val dateFormat = if (locale.language == "zh") {
                SimpleDateFormat("M月d日", locale)
            } else {
                SimpleDateFormat("MMM d", locale) // For example: Jun 20, Dec 15
            }
            dateFormat.format(chatDate)
        }
        
        tvTimeInfo.text = formattedTime
        tvTimeInfo.visibility = View.VISIBLE
    }
    
    private fun isSameDay(date1: Date, date2: Date): Boolean {
        val cal1 = java.util.Calendar.getInstance()
        val cal2 = java.util.Calendar.getInstance()
        cal1.time = date1
        cal2.time = date2
        return cal1.get(java.util.Calendar.YEAR) == cal2.get(java.util.Calendar.YEAR) &&
                cal1.get(java.util.Calendar.DAY_OF_YEAR) == cal2.get(java.util.Calendar.DAY_OF_YEAR)
    }

    private fun getFormattedFileSize(modelWrapper: ModelListManager.ModelItemWrapper): String {
        val modelItem = modelWrapper.modelItem
        
        // Try to get file size using the same method as MarketItemHolder
        modelItem.modelId?.let { modelId ->
            val downloadedFile = modelDownloadManager.getDownloadedFile(modelId)
            if (downloadedFile != null) {
                return FileUtils.getFileSizeString(downloadedFile)
            }
        }
        
        // Fallback to direct file size calculation
        if (modelWrapper.downloadSize > 0) {
            return FileUtils.formatFileSize(modelWrapper.downloadSize)
        }
        
        // Try to get size from local path
        modelItem.localPath?.let { localPath ->
            val file = File(localPath)
            if (file.exists()) {
                return FileUtils.getFileSizeString(file)
            }
        }
        
        return ""
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

    private fun getDisplayTags(modelItem: ModelItem): List<String> {
        val tags = mutableListOf<String>()
        
        // Add source tag first
        val source = getModelSource(modelItem.modelId)
        if (source != null) {
            tags.add(source)
        }
        
        // Use getTags() which now prioritizes market tags from model_market.json
        val marketTags = modelItem.getTags()
        
        // Add local/downloaded status
        if (modelItem.isLocal) {
            tags.add(itemView.context.getString(R.string.local))
        } else if (marketTags.isNotEmpty()) {
            // If we have market tags, use them directly (they're already user-friendly)
            tags.addAll(marketTags.take(2)) // Limit to 2 market tags to leave room for source tag
        }
        
        // Limit total tags to 3 for better UI layout
        return tags.take(3)
    }

    fun bind(modelWrapper: ModelListManager.ModelItemWrapper) {
        val modelItem = modelWrapper.modelItem
        val modelName = modelItem.modelName
        
        // Store current wrapper and set item tag
        this.currentModelWrapper = modelWrapper
        itemView.tag = modelWrapper
        
        // Set basic model info
        tvModelTitle.text = modelName
        headerSection.setModelName(modelName)
        
        // Use consistent tag display logic
        tagsLayout.setTags(getDisplayTags(modelItem))
        
        // Display time information from wrapper
        displayTimeInfo(modelWrapper)
        
        // Show pinned overlay
        pinnedOverlay.visibility = if (modelWrapper.isPinned || true) View.VISIBLE else View.GONE
        
        // Use consistent file size display logic
        val formattedSize = getFormattedFileSize(modelWrapper)
        tvStatus.text = if (formattedSize.isNotEmpty()) {
            tvStatus.resources.getString(R.string.downloaded_click_to_chat, formattedSize)
        } else {
            tvStatus.resources.getString(R.string.downloaded_click_to_chat, "")
        }
        
        Log.d(TAG, "itemView id : ${itemView.id == R.id.recycler_item_model_parent}")
        itemView.isActivated = modelWrapper.isPinned
    }

    override fun onClick(v: View) {
        val modelWrapper = v.tag as ModelListManager.ModelItemWrapper
        modelItemListener.onItemClicked(modelWrapper.modelItem)
    }

    override fun onLongClick(v: View): Boolean {
        if (!enableLongClick) {
            return false
        }
        
        val modelWrapper = itemView.tag as ModelListManager.ModelItemWrapper
        val modelItem = modelWrapper.modelItem
        
        val popupMenu = PopupMenu(v.context, tvStatus)
        val inflater = popupMenu.menuInflater
        inflater.inflate(R.menu.model_list_item_context_menu, popupMenu.menu)
        
        popupMenu.menu.add(0, R.id.menu_pin_model, 0,
            if (modelWrapper.isPinned) R.string.menu_unpin_model else R.string.menu_pin_model)
        
        popupMenu.setOnMenuItemClickListener { item: MenuItem ->
            val modelId = modelItem.modelId
            if (item.itemId == R.id.menu_delete_model) {
                MaterialAlertDialogBuilder(v.context)
                    .setTitle(R.string.confirm_delete_model_title)
                    .setMessage(R.string.confirm_delete_model_message)
                    .setPositiveButton(android.R.string.ok) { _, _ ->
                        MainScope().launch {
                            try {
                                ModelDownloadManager.getInstance(v.context).deleteModel(modelId!!)
                                // Notify the listener that the model was deleted successfully
                                modelItemListener.onItemDeleted(modelItem)
                            } catch (e: Exception) {
                                Log.e(TAG, "Failed to delete model: $modelId", e)
                                // You could show an error toast here if needed
                            }
                        }
                    }
                    .setNegativeButton(android.R.string.cancel, null)
                    .show()
            } else if (item.itemId == R.id.menu_settings) {
                val context = v.context
                val modelId = modelItem.modelId
                if (ModelUtils.isDiffusionModel(modelId!!)) {
                    Toast.makeText(context, R.string.diffusion_model_not_alloed, Toast.LENGTH_SHORT).show()
                    return@setOnMenuItemClickListener true
                }
                val fragmentManager = (context as? AppCompatActivity)?.supportFragmentManager
                if (fragmentManager != null) {
                    val settingsSheet = SettingsBottomSheetFragment()
                    settingsSheet.setModelId(modelId)
                    settingsSheet.setConfigPath(modelItem.localPath)
                    settingsSheet.show(fragmentManager, SettingsBottomSheetFragment.TAG)
                }
            } else if (item.itemId == R.id.menu_show_model_info) {
                // Show model info (simplified version)
                val info = StringBuilder()
                info.append("Model: ${modelItem.modelName ?: modelItem.modelId}\n")
                val sizeInfo = currentModelWrapper?.let { getFormattedFileSize(it) } ?: "Unknown"
                info.append("Size: ${if (sizeInfo.isNotEmpty()) sizeInfo else "Unknown"}\n")
                if ((currentModelWrapper?.lastChatTime ?: 0) > 0) {
                    val dateFormat = SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.getDefault())
                    info.append("Last Chat: ${dateFormat.format(Date(currentModelWrapper!!.lastChatTime))}\n")
                }
                if ((currentModelWrapper?.downloadTime ?: 0) > 0) {
                    val dateFormat = SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.getDefault())
                    info.append("Downloaded: ${dateFormat.format(Date(currentModelWrapper!!.downloadTime))}")
                }
                
                AlertDialog.Builder(v.context)
                    .setTitle("Model Information")
                    .setMessage(info.toString())
                    .setPositiveButton(android.R.string.ok, null)
                    .show()
            } else if (item.itemId == R.id.menu_open_model_card) {
                ModelUtils.openModelCard(v.context, modelItem)
            } else if (item.itemId == R.id.menu_pin_model) {
                // Handle pin/unpin action
                return@setOnMenuItemClickListener modelItemListener.onItemLongClicked(modelItem)
            }
            true
        }
        
        // Since all models are downloaded, simplify menu visibility
        // Delete, settings, and model info are always available
        popupMenu.menu.findItem(R.id.menu_delete_model).setVisible(!modelItem.isLocal)
        popupMenu.menu.findItem(R.id.menu_settings).setVisible(true)
        popupMenu.menu.findItem(R.id.menu_show_model_info).setVisible(false)
        
        // Download control items are not needed for downloaded models
        popupMenu.menu.findItem(R.id.menu_pause_download)?.setVisible(false)
        popupMenu.menu.findItem(R.id.menu_start_download)?.setVisible(false)
        
        // Model card should be visible for remote models
        popupMenu.menu.findItem(R.id.menu_open_model_card).setVisible(!modelItem.isLocal)
        
        popupMenu.show()
        return true
    }

    companion object {
        const val TAG = "ModelItemHolder"
    }
}