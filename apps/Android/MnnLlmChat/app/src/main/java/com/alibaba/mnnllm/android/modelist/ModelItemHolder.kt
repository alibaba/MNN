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
import com.alibaba.mls.api.download.DownloadInfo
import com.alibaba.mls.api.download.DownloadState
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.model.ModelTypeUtils
import com.alibaba.mnnllm.android.model.ModelUtils
import com.alibaba.mnnllm.android.modelsettings.SettingsBottomSheetFragment
import com.alibaba.mnnllm.android.utils.DialogUtils
import com.alibaba.mnnllm.android.utils.FileUtils
import com.alibaba.mnnllm.android.widgets.ModelAvatarView
import com.alibaba.mnnllm.android.widgets.TagsLayout
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
    private var tvStatus: TextView
    private var tvTimeInfo: TextView
    private val btnUpdate: com.google.android.material.button.MaterialButton

    private val headerSection: ModelAvatarView
    private val tagsLayout: TagsLayout
    private val pinnedOverlay: View // Pinned overlay

    private var currentModelWrapper: ModelItemWrapper? = null
    private val modelDownloadManager = ModelDownloadManager.getInstance(itemView.context)

    init {
        itemView.setOnClickListener(this)
        if (enableLongClick) {
            itemView.setOnLongClickListener(this)
        }
        tvModelName = itemView.findViewById(R.id.tvModelName)
        tvModelTitle = itemView.findViewById(R.id.tvModelTitle)
        tvStatus = itemView.findViewById(R.id.tvStatus)
        tvTimeInfo = itemView.findViewById(R.id.tvTimeInfo)
        btnUpdate = itemView.findViewById(R.id.btn_update)
        headerSection = itemView.findViewById(R.id.header_section_title)
        tagsLayout = itemView.findViewById(R.id.tagsLayout)
        pinnedOverlay = itemView.findViewById(R.id.pinned_overlay)
        
        // Set update button click listener
        btnUpdate.setOnClickListener {
            currentModelWrapper?.let { wrapper ->
                modelItemListener.onItemUpdate(wrapper.modelItem)
            }
        }
    }

    private fun displayTimeInfo(modelWrapper: ModelItemWrapper) {
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

    private fun getFormattedFileSize(modelWrapper: ModelItemWrapper): String {
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
        return com.alibaba.mnnllm.android.modelmarket.TagMapper.getDisplayTagList(modelItem.tags).take(3)
    }

    fun bind(modelWrapper: ModelItemWrapper) {
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
        pinnedOverlay.visibility = if (modelWrapper.isPinned) View.VISIBLE else View.GONE
        
        // Handle update button visibility and status
        updateButtonAndStatus(modelWrapper)
        
        itemView.isActivated = modelWrapper.isPinned
    }

    /**
     * Update the update button visibility and status text based on model state
     */
    private fun updateButtonAndStatus(modelWrapper: ModelItemWrapper) {
        val formattedSize = getFormattedFileSize(modelWrapper)
        
        // Check if model is currently updating (hasUpdate and downloading)
        val isUpdating = isModelUpdating(modelWrapper)
        
        if (modelWrapper.hasUpdate) {
            btnUpdate.visibility = View.VISIBLE
            if (isUpdating) {
                btnUpdate.text = btnUpdate.resources.getString(R.string.download_state_updating)
                btnUpdate.isEnabled = false
                tvStatus.text = if (formattedSize.isNotEmpty()) {
                    "${formattedSize} (${tvStatus.resources.getString(R.string.download_state_updating)})"
                } else {
                    tvStatus.resources.getString(R.string.download_state_updating)
                }
            } else {
                btnUpdate.text = btnUpdate.resources.getString(R.string.update)
                btnUpdate.isEnabled = true
                tvStatus.text = if (formattedSize.isNotEmpty()) {
                    tvStatus.resources.getString(R.string.downloaded_update_available, formattedSize)
                } else {
                    tvStatus.resources.getString(R.string.downloaded_update_available, "")
                }
            }
        } else {
            btnUpdate.visibility = View.GONE
            tvStatus.text = if (formattedSize.isNotEmpty()) {
                tvStatus.resources.getString(R.string.downloaded_click_to_chat, formattedSize)
            } else {
                tvStatus.resources.getString(R.string.downloaded_click_to_chat, "")
            }
        }
    }

    /**
     * Check if the model is currently being updated
     * @param modelWrapper The model wrapper to check
     * @return true if model hasUpdate and is currently downloading
     */
    private fun isModelUpdating(modelWrapper: ModelItemWrapper): Boolean {
        if (!modelWrapper.hasUpdate) return false
        
        modelWrapper.modelItem.modelId?.let { modelId ->
            val downloadInfo = modelDownloadManager.getDownloadInfo(modelId)
            return downloadInfo.downloadState == DownloadState.DOWNLOADING
        }
        return false
    }

    /**
     * Update progress for the model if it's currently being updated
     * This method should be called from the adapter when progress updates are received
     */
    fun updateProgress(downloadInfo: DownloadInfo) {
        currentModelWrapper?.let { wrapper ->
            // If model has update and is downloading, refresh the UI
            if (wrapper.hasUpdate && downloadInfo.downloadState == DownloadState.DOWNLOADING) {
                updateButtonAndStatus(wrapper)
            }
        }
    }

    override fun onClick(v: View) {
        val modelWrapper = v.tag as ModelItemWrapper
        modelItemListener.onItemClicked(modelWrapper.modelItem)
    }

    override fun onLongClick(v: View): Boolean {
        if (!enableLongClick) {
            return false
        }
        
        val modelWrapper = itemView.tag as ModelItemWrapper
        val modelItem = modelWrapper.modelItem
        
        val popupMenu = PopupMenu(v.context, tvStatus)
        val inflater = popupMenu.menuInflater
        inflater.inflate(R.menu.model_list_item_context_menu, popupMenu.menu)
        
        popupMenu.menu.add(0, R.id.menu_pin_model, 0,
            if (modelWrapper.isPinned) R.string.menu_unpin_model else R.string.menu_pin_model)
        
        popupMenu.setOnMenuItemClickListener { item: MenuItem ->
            val modelId = modelItem.modelId
            if (item.itemId == R.id.menu_delete_model) {
                DialogUtils.showDeleteConfirmationDialog(v.context) {
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
            } else if (item.itemId == R.id.menu_settings) {
                val context = v.context
                val modelId = modelItem.modelId
                if (ModelTypeUtils.isDiffusionModel(modelId!!)) {
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
                // Show model info directly
                val context = v.context
                val info = StringBuilder()
                
                info.append("Model Name: ${modelItem.modelName ?: modelItem.modelId}\n\n")
                
                // Show storage path
                val storagePath = modelItem.localPath
                info.append("Storage Location:\n$storagePath\n\n")
                
                // Show size
                val sizeInfo = currentModelWrapper?.let { getFormattedFileSize(it) } ?: "Unknown"
                info.append("Size: ${if (sizeInfo.isNotEmpty()) sizeInfo else "Unknown"}\n")
                
                // Show last chat time
                if ((currentModelWrapper?.lastChatTime ?: 0) > 0) {
                    val dateFormat = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault())
                    info.append("Last Chat Time: ${dateFormat.format(Date(currentModelWrapper!!.lastChatTime))}\n")
                } else {
                    info.append("Last Chat Time: Never\n")
                }
                
                if ((currentModelWrapper?.downloadTime ?: 0) > 0) {
                    val dateFormat = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault())
                    info.append("Downloaded Time: ${dateFormat.format(Date(currentModelWrapper!!.downloadTime))}\n")
                }
                
                AlertDialog.Builder(context)
                    .setTitle(R.string.menu_show_model_info_title)
                    .setMessage(info.toString())
                    .setPositiveButton(android.R.string.ok, null)
                    .show()
            } else if (item.itemId == R.id.menu_update_model) {
                // Handle update action
                modelItemListener.onItemUpdate(modelItem)
            } else if (item.itemId == R.id.menu_open_model_card) {
                ModelUtils.openModelCard(v.context, modelItem)
            } else if (item.itemId == R.id.menu_pin_model) {
                // Handle pin/unpin action
                return@setOnMenuItemClickListener modelItemListener.onItemLongClicked(modelItem)
            }
            true
        }
        
        // Determine if model is deletable
        // Deletable condition:
        // 1. Must be a downloaded model (has download info) -> This excludes manually added local models
        // 2. Must NOT be a builtin model -> This excludes builtin models
        val isDownloaded = modelWrapper.downloadedModelInfo != null
        val isBuiltin = modelItem.isBuiltin || modelItem.modelId?.startsWith("Builtin/") == true
        val isDeletable = isDownloaded && !isBuiltin

        // Delete, settings, and model info are always available
        popupMenu.menu.findItem(R.id.menu_delete_model).setVisible(isDeletable)
        popupMenu.menu.findItem(R.id.menu_settings).setVisible(true)
        // Show model info menu item always
        popupMenu.menu.findItem(R.id.menu_show_model_info).setVisible(true)
        
        // Show update option only for remote models with updates
        popupMenu.menu.findItem(R.id.menu_update_model).setVisible(isDownloaded && modelWrapper.hasUpdate)
        
        // Download control items are not needed for downloaded models
        popupMenu.menu.findItem(R.id.menu_pause_download)?.setVisible(false)
        popupMenu.menu.findItem(R.id.menu_start_download)?.setVisible(false)
        
        // Model card should be visible for remote models
        popupMenu.menu.findItem(R.id.menu_open_model_card)?.setVisible(false)
        
        popupMenu.show()
        return true
    }

    companion object {
        const val TAG = "ModelItemHolder"
    }
}