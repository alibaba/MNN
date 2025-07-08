package com.alibaba.mnnllm.android.modelmarket

import android.annotation.SuppressLint
import android.text.TextUtils
import android.util.Log
import android.view.MenuItem
import android.view.View
import android.view.View.OnLongClickListener
import com.google.android.material.button.MaterialButton
import android.widget.CheckBox
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.PopupMenu
import androidx.recyclerview.widget.RecyclerView
import com.alibaba.mls.api.ModelItem
import com.alibaba.mls.api.download.DownloadInfo
import com.alibaba.mls.api.download.DownloadState
import com.alibaba.mls.api.download.ModelDownloadManager
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.model.ModelUtils
import com.alibaba.mnnllm.android.utils.FileUtils
import com.alibaba.mnnllm.android.modelsettings.SettingsBottomSheetFragment
import com.alibaba.mnnllm.android.widgets.ModelAvatarView
import com.alibaba.mnnllm.android.widgets.TagsLayout


class MarketItemHolder(
    itemView: View, 
    private val modelMarketItemListener: ModelMarketItemListener,
    private val enableLongClick: Boolean = true
) : RecyclerView.ViewHolder(itemView), View.OnClickListener, OnLongClickListener {

    // UI Components
    private val tvModelTitle: TextView = itemView.findViewById(R.id.tvModelTitle)
    private val tvStatus: TextView = itemView.findViewById(R.id.tvStatus)
    private val headerSection: ModelAvatarView = itemView.findViewById(R.id.header_section_title)
    private val tagsLayout: TagsLayout = itemView.findViewById(R.id.tagsLayout)
    private val btnDownloadAction: MaterialButton = itemView.findViewById(R.id.btn_download_action)
    private val checkboxVoiceModel: CheckBox = itemView.findViewById(R.id.checkbox_voice_model)

    // Data
    private var modelMarketItemWrapper: ModelMarketItemWrapper? = null
    private val modelDownloadManager = ModelDownloadManager.getInstance(itemView.context)
    
    // Voice model delegate
    private val voiceDelegate = MarketHolderVoiceDelegate(itemView.context, modelDownloadManager)

    init {
        itemView.setOnClickListener(this)
        if (enableLongClick) {
            itemView.setOnLongClickListener(this)
        }
        btnDownloadAction.setOnClickListener {
            modelMarketItemWrapper?.let { wrapper ->
                modelMarketItemListener.onActionClicked(wrapper)
            }
        }
        
        // CheckBox listener will be set in updateDownloadState when needed
    }

    /**
     * Set callback to be executed when a voice model is set as default
     */
    fun setVoiceModelChangedCallback(callback: (MarketHolderVoiceDelegate.VoiceModelType, String) -> Unit) {
        voiceDelegate.setOnVoiceModelChangedCallback(callback)
    }

    fun bind(modelMarketItemWrapper: ModelMarketItemWrapper) {
        this.modelMarketItemWrapper = modelMarketItemWrapper
        itemView.tag = modelMarketItemWrapper

        val modelMarketItem = modelMarketItemWrapper.modelMarketItem
        
        // Set basic info
        tvModelTitle.text = modelMarketItem.modelName
        tagsLayout.setTags(modelMarketItem.tags.take(3)) // Limit to 3 tags to prevent overcrowding
        headerSection.setModelName(modelMarketItem.modelName)
        
        // Update download state
        updateDownloadState(modelMarketItemWrapper.downloadInfo)
    }

    @SuppressLint("DefaultLocale")
    fun updateProgress(downloadInfo: DownloadInfo) {
        Log.d(TAG, "updateProgress called for model: ${(itemView.tag as? ModelMarketItemWrapper)?.modelMarketItem?.modelId}, progress: ${downloadInfo.progress}")
        
        // Update internal state
        modelMarketItemWrapper?.downloadInfo = downloadInfo
        
        // Update UI
        updateDownloadState(downloadInfo)
    }

    @SuppressLint("DefaultLocale")
    private fun updateDownloadState(downloadInfo: DownloadInfo) {
        val downloadState = downloadInfo.downloadState
        
        when (downloadState) {
            DownloadState.NOT_START, DownloadState.FAILED -> {
                btnDownloadAction.visibility = View.VISIBLE
                checkboxVoiceModel.visibility = View.GONE
                btnDownloadAction.text = btnDownloadAction.resources.getString(R.string.download)
                setButtonStyle()
                updateStatusText(downloadInfo)
            }
            
            DownloadState.DOWNLOADING -> {
                btnDownloadAction.visibility = View.VISIBLE
                checkboxVoiceModel.visibility = View.GONE
                if (TextUtils.equals("Preparing", downloadInfo.progressStage)) {
                    btnDownloadAction.text = btnDownloadAction.resources.getString(R.string.download_pending)
                    setButtonStyle()
                    tvStatus.text = tvStatus.resources.getString(R.string.download_preparing)
                } else {
                    val progressText = String.format("%.2f%%", downloadInfo.progress * 100)
                    btnDownloadAction.text = progressText
                    setButtonStyle()
                    updateProgressText(downloadInfo)
                }
            }
            
            DownloadState.PAUSED -> {
                btnDownloadAction.visibility = View.VISIBLE
                checkboxVoiceModel.visibility = View.GONE
                btnDownloadAction.text = btnDownloadAction.resources.getString(R.string.download_resume)
                setButtonStyle()
                updateStatusText(downloadInfo)
            }
            
            DownloadState.COMPLETED -> {
                val modelMarketItem = modelMarketItemWrapper?.modelMarketItem
                if (modelMarketItem != null) {
                    val voiceType = voiceDelegate.getVoiceModelType(modelMarketItem)
                    if (voiceType != MarketHolderVoiceDelegate.VoiceModelType.NONE) {
                        // Handle voice model (TTS/ASR) - show checkbox
                        val isDefault = voiceDelegate.isDefaultModel(modelMarketItem.modelId, voiceType)
                        voiceDelegate.setVoiceModelUI(
                            btnDownloadAction, 
                            checkboxVoiceModel, 
                            true, 
                            isDefault, 
                            voiceType, 
                            modelMarketItem
                        ) {
                            // Update UI when model is changed
                            updateDownloadState(downloadInfo)
                            
                            // Refresh the entire adapter to update other models' default status
                            modelMarketItemWrapper?.let {
                                modelMarketItemListener.onDefaultVoiceModelChanged(it)
                            }
                        }
                        tvStatus.text = voiceDelegate.getVoiceModelStatusText(downloadInfo, modelMarketItem.modelId, isDefault, voiceType)
                    } else {
                        // Handle regular model - show chat button
                        btnDownloadAction.visibility = View.VISIBLE
                        checkboxVoiceModel.visibility = View.GONE
                        btnDownloadAction.text = btnDownloadAction.resources.getString(R.string.chat_action)
                        setButtonStyle()
                        updateStatusText(downloadInfo)
                    }
                } else {
                    btnDownloadAction.visibility = View.VISIBLE
                    checkboxVoiceModel.visibility = View.GONE
                    btnDownloadAction.text = btnDownloadAction.resources.getString(R.string.chat_action)
                    setButtonStyle()
                    updateStatusText(downloadInfo)
                }
            }
            
            else -> {
                btnDownloadAction.visibility = View.VISIBLE
                checkboxVoiceModel.visibility = View.GONE
                btnDownloadAction.text = btnDownloadAction.resources.getString(R.string.download)
                setButtonStyle()
            }
        }
    }
    
    private fun setButtonStyle() {
        // 所有状态都使用相同的边框样式和主题色
        val context = itemView.context
        val typedValue = android.util.TypedValue()
        context.theme.resolveAttribute(com.google.android.material.R.attr.colorPrimary, typedValue, true)
        val primaryColor = typedValue.data
        
        // 清除图标（确保非TTS完成状态不显示图标）
        btnDownloadAction.icon = null
        btnDownloadAction.setTextColor(primaryColor)
        btnDownloadAction.strokeColor = android.content.res.ColorStateList.valueOf(primaryColor)
    }

    @SuppressLint("DefaultLocale")
    private fun updateStatusText(downloadInfo: DownloadInfo) {
        val modelMarketItem = modelMarketItemWrapper?.modelMarketItem ?: return
        
        tvStatus.text = when (downloadInfo.downloadState) {
            DownloadState.NOT_START -> {
                tvStatus.resources.getString(
                    R.string.download_not_started,
                    if (downloadInfo.totalSize > 0) {
                        FileUtils.formatFileSize(downloadInfo.totalSize)
                    } else ""
                )
            }
            
            DownloadState.COMPLETED -> {
                tvStatus.resources.getString(
                    R.string.downloaded_click_to_chat,
                    FileUtils.getFileSizeString(modelDownloadManager.getDownloadedFile(modelMarketItem.modelId))
                )
            }
            
            DownloadState.FAILED -> {
                Log.d(TAG, "Binding FAILED state for model: ${modelMarketItem.modelId}")
                tvStatus.resources.getString(
                    R.string.download_failed_click_retry,
                    downloadInfo.errorMessage
                )
            }
            
            DownloadState.PAUSED -> {
                val pausedText = tvStatus.resources.getString(R.string.downloading_paused, downloadInfo.progress * 100)
                if (downloadInfo.totalSize > 0) {
                    "${FileUtils.formatFileSize(downloadInfo.totalSize)} | $pausedText"
                } else {
                    pausedText
                }
            }
            
            else -> tvStatus.text
        }
    }

    private fun updateProgressText(downloadInfo: DownloadInfo) {
        tvStatus.text = itemView.resources.getString(
            R.string.downloading_progress,
            if (downloadInfo.totalSize > 0) {
                FileUtils.formatFileSize(downloadInfo.totalSize)
            } else "",
            downloadInfo.speedInfo
        )
    }

    override fun onClick(v: View) {
        val modelMarketItemWrapper = v.tag as ModelMarketItemWrapper
        val modelMarketItem = modelMarketItemWrapper.modelMarketItem
        
        // 如果是下载完成的语音模型，并且CheckBox可见，则触发CheckBox逻辑
        if (modelMarketItemWrapper.downloadInfo.downloadState == DownloadState.COMPLETED) {
            val voiceType = voiceDelegate.getVoiceModelType(modelMarketItem)
            if (voiceType != MarketHolderVoiceDelegate.VoiceModelType.NONE && checkboxVoiceModel.visibility == View.VISIBLE) {
                // 切换CheckBox状态，让OnCheckedChangeListener处理逻辑
                val newCheckedState = !checkboxVoiceModel.isChecked
                checkboxVoiceModel.isChecked = newCheckedState
                return
            }
        }
        
        // 对于非语音模型或未下载完成的模型，执行默认逻辑
        modelMarketItemListener.onActionClicked(modelMarketItemWrapper)
    }



    override fun onLongClick(v: View): Boolean {
        if (!enableLongClick) {
            return false
        }
        
        val modelMarketItemWrapper = itemView.tag as ModelMarketItemWrapper
        val modelMarketItem = modelMarketItemWrapper.modelMarketItem
        
        showContextMenu(modelMarketItemWrapper, modelMarketItem)
        return true
    }

    private fun showContextMenu(modelMarketItemWrapper: ModelMarketItemWrapper, modelMarketItem: ModelMarketItem) {
        val popupMenu = PopupMenu(itemView.context, tvStatus)
        popupMenu.menuInflater.inflate(R.menu.market_item_context_menu, popupMenu.menu)
        
        setupMenuClickListener(popupMenu, modelMarketItemWrapper, modelMarketItem)
        configureMenuVisibility(popupMenu, modelMarketItemWrapper.downloadInfo.downloadState)
        
        popupMenu.show()
    }

    private fun setupMenuClickListener(
        popupMenu: PopupMenu,
        modelMarketItemWrapper: ModelMarketItemWrapper,
        modelMarketItem: ModelMarketItem
    ) {
        popupMenu.setOnMenuItemClickListener { item: MenuItem ->
            when (item.itemId) {
                R.id.menu_delete_model -> {
                    modelMarketItemListener.onDeleteClicked(modelMarketItemWrapper)
                }
                R.id.menu_pause_download -> {
                    Log.d(TAG, "pauseDownload ${modelMarketItem.modelId}")
                    modelMarketItemListener.onPauseClicked(modelMarketItemWrapper)
                }
                R.id.menu_start_download -> {
                    modelMarketItemListener.onDownloadOrResumeClicked(modelMarketItemWrapper)
                }
                R.id.menu_settings -> {
                    handleSettingsMenu(modelMarketItem)
                }
                R.id.menu_open_model_card -> {
                    openModelCard(itemView.context, modelMarketItem)
                }
            }
            true
        }
    }

    private fun configureMenuVisibility(popupMenu: PopupMenu, downloadState: Int) {
        val menu = popupMenu.menu
        
        // Delete option: visible for completed, paused, or failed downloads
        menu.findItem(R.id.menu_delete_model).isVisible = 
            downloadState in listOf(DownloadState.COMPLETED, DownloadState.PAUSED, DownloadState.FAILED)
        
        // Pause option: visible only when downloading
        menu.findItem(R.id.menu_pause_download).isVisible = 
            downloadState == DownloadState.DOWNLOADING
        
        // Start/Resume option: visible for not started, paused, or failed downloads
        menu.findItem(R.id.menu_start_download).isVisible = 
            downloadState in listOf(DownloadState.PAUSED, DownloadState.NOT_START, DownloadState.FAILED)
        
        // Settings option: visible only for completed downloads
        menu.findItem(R.id.menu_settings).isVisible = 
            downloadState == DownloadState.COMPLETED
        
        // Model card: always visible
        menu.findItem(R.id.menu_open_model_card).isVisible = true
    }

    private fun handleSettingsMenu(modelMarketItem: ModelMarketItem) {
        val context = itemView.context
        if (ModelUtils.isDiffusionModel(modelMarketItem.modelName)) {
            Toast.makeText(context, R.string.diffusion_model_not_alloed, Toast.LENGTH_SHORT).show()
            return
        }
        
        val fragmentManager = (context as? AppCompatActivity)?.supportFragmentManager
        if (fragmentManager != null) {
            val settingsSheet = SettingsBottomSheetFragment()
            settingsSheet.setModelId(modelMarketItem.modelId)
            settingsSheet.setConfigPath(null) // ModelMarketItem doesn't have localPath
            settingsSheet.show(fragmentManager, SettingsBottomSheetFragment.TAG)
        }
    }

    private fun openModelCard(context: android.content.Context, modelMarketItem: ModelMarketItem) {
        // Create a ModelItem for compatibility with ModelUtils.openModelCard
        val modelItem = ModelItem.fromDownloadModel(modelMarketItem.modelId, "")
        ModelUtils.openModelCard(context, modelItem)
    }

    companion object {
        const val TAG = "ModelItemHolder"
    }
}