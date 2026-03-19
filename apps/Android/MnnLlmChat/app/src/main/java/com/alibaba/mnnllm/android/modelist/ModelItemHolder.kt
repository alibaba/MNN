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
import com.alibaba.mnnllm.android.modelsettings.DiffusionSettingsBottomSheetFragment
import com.alibaba.mnnllm.android.utils.DialogUtils
import com.alibaba.mnnllm.android.widgets.ModelAvatarView
import com.alibaba.mnnllm.android.widgets.TagsLayout
import kotlinx.coroutines.MainScope
import kotlinx.coroutines.launch
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

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

    fun bind(modelWrapper: ModelItemWrapper) {
        val modelItem = modelWrapper.modelItem

        // Store current wrapper and set item tag
        this.currentModelWrapper = modelWrapper
        itemView.tag = modelWrapper

        render(
            ModelListItemUiStateFactory.fromModelWrapper(
                context = itemView.context,
                modelWrapper = modelWrapper,
                modelDownloadManager = modelDownloadManager
            )
        )
        itemView.isActivated = modelWrapper.isPinned
    }

    internal fun render(uiState: ModelListItemUiState) {
        tvModelTitle.text = uiState.title
        headerSection.setModelName(uiState.title)
        tagsLayout.setTags(uiState.tags)

        tvStatus.text = uiState.statusText

        if (uiState.timeText.isNullOrEmpty()) {
            tvTimeInfo.visibility = View.GONE
        } else {
            tvTimeInfo.text = uiState.timeText
            tvTimeInfo.visibility = View.VISIBLE
        }

        btnUpdate.visibility = if (uiState.updateButtonVisible) View.VISIBLE else View.GONE
        btnUpdate.text = uiState.updateButtonText
        btnUpdate.isEnabled = uiState.updateButtonEnabled
        pinnedOverlay.visibility = if (uiState.isPinned) View.VISIBLE else View.GONE
        itemView.isActivated = uiState.isPinned
    }

    /**
     * Update progress for the model if it's currently being updated
     * This method should be called from the adapter when progress updates are received
     */
    fun updateProgress(downloadInfo: DownloadInfo) {
        currentModelWrapper?.let { wrapper ->
            // If model has update and is downloading, refresh the UI
            if (wrapper.hasUpdate && downloadInfo.downloadState == DownloadState.DOWNLOADING) {
                render(
                    ModelListItemUiStateFactory.fromModelWrapper(
                        context = itemView.context,
                        modelWrapper = wrapper,
                        modelDownloadManager = modelDownloadManager
                    )
                )
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
                            // Use ModelDeletionHelper for proper cleanup of mmap cache and chat sessions
                            val result = ModelDeletionHelper.deleteModelWithCleanup(v.context, modelId!!)
                            if (!result.modelDeleted) {
                                Log.w(TAG, "Model deletion incomplete: ${result.errors.joinToString()}")
                            }
                            // Notify the listener that the model was deleted successfully
                            modelItemListener.onItemDeleted(modelItem)
                        } catch (e: Exception) {
                            Log.e(TAG, "Failed to delete model: $modelId", e)
                        }
                    }
                }
            } else if (item.itemId == R.id.menu_settings) {
                val context = v.context
                val modelId = modelItem.modelId
                val fragmentManager = (context as? AppCompatActivity)?.supportFragmentManager
                if (fragmentManager != null) {
                    if (ModelTypeUtils.isDiffusionModel(modelId!!)) {
                        val settingsSheet = DiffusionSettingsBottomSheetFragment()
                        settingsSheet.setModelId(modelId)
                        settingsSheet.setConfigPath(ModelUtils.getConfigPathForModel(modelItem))
                        settingsSheet.show(fragmentManager, DiffusionSettingsBottomSheetFragment.TAG)
                    } else {
                        val settingsSheet = SettingsBottomSheetFragment()
                        settingsSheet.setModelId(modelId)
                        settingsSheet.setConfigPath(ModelUtils.getConfigPathForModel(modelItem))
                        settingsSheet.show(fragmentManager, SettingsBottomSheetFragment.TAG)
                    }
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
                val sizeInfo = currentModelWrapper?.let {
                    ModelListItemUiStateFactory.getFormattedFileSize(it, modelDownloadManager)
                } ?: "Unknown"
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
