package com.alibaba.mnnllm.android.modelist

import android.annotation.SuppressLint
import android.text.TextUtils
import android.view.MenuItem
import android.view.View
import android.view.View.OnLongClickListener
import android.widget.ImageView
import android.widget.ProgressBar
import android.widget.TextView
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.widget.PopupMenu
import androidx.recyclerview.widget.RecyclerView
import com.alibaba.mls.api.ModelItem
import com.alibaba.mls.api.download.DownloadInfo
import com.alibaba.mls.api.download.ModelDownloadManager
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.utils.FileUtils
import com.alibaba.mnnllm.android.utils.ModelUtils.getDrawableId
import com.alibaba.mnnllm.android.widgets.TagsLayout
import kotlinx.coroutines.MainScope
import kotlinx.coroutines.launch
import java.io.File

class ModelItemHolder(itemView: View, private val modelItemListener: ModelItemListener) :
    RecyclerView.ViewHolder(itemView), View.OnClickListener, OnLongClickListener {
    private var tvModelName: TextView
    private var tvModelTitle: TextView
    private var tvModelSubtitle: TextView
    private var tvStatus: TextView
    private var headerIcon: ImageView

    private val headerSection: View

    private var downloadProgressView: View
    private var progressBar: ProgressBar

    private var modelItemDownloadState: ModelItemDownloadState? = null

    private val tagsLayout: TagsLayout

    private val iconDownload:View
    private val modelDownloadManager = ModelDownloadManager.getInstance(itemView.context)

    init {
        itemView.setOnClickListener(this)
        itemView.setOnLongClickListener(this)
        tvModelName = itemView.findViewById(R.id.tvModelName)
        tvModelTitle = itemView.findViewById(R.id.tvModelTitle)
        tvModelSubtitle = itemView.findViewById(R.id.tvModelSubtitle)
        tvStatus = itemView.findViewById(R.id.tvStatus)
        headerSection = itemView.findViewById(R.id.header_section_title)
        headerIcon = itemView.findViewById(R.id.header_section_icon)
        downloadProgressView = itemView.findViewById(R.id.download_progress_view)
        tagsLayout = itemView.findViewById(R.id.tagsLayout)
        progressBar = itemView.findViewById(R.id.download_progress_bar)
        iconDownload = itemView.findViewById(R.id.iv_download)
    }

    fun bind(hfModelItem: ModelItem, modelItemDownloadState: ModelItemDownloadState?) {
        val modelName = hfModelItem.modelName
        itemView.tag = hfModelItem
        this.modelItemDownloadState = modelItemDownloadState
        tvModelTitle.text = modelName
        tagsLayout.setTags(
            hfModelItem.newTags
        )
        val drawableId = getDrawableId(modelName)
        if (drawableId != 0) {
            headerIcon.visibility = View.VISIBLE
            headerIcon.setImageResource(drawableId)
            tvModelName.visibility = View.INVISIBLE
        } else {
            headerIcon.visibility = View.INVISIBLE
            val headerText = modelName?.replace("_", "-") ?: ""
            tvModelName.text =
                if (headerText.contains("-")) headerText.substring(
                    0,
                    headerText.indexOf("-")
                ) else headerText
            tvModelName.visibility = View.VISIBLE
        }
        if (modelItemDownloadState == null) {
            progressBar.visibility = View.GONE
            tvStatus.text = ""
            return
        }
        val downloadState = modelItemDownloadState.downloadInfo!!.downlodaState
        iconDownload.visibility = if (downloadState != DownloadInfo.DownloadSate.PAUSED && downloadState != DownloadInfo.DownloadSate.NOT_START && downloadState != DownloadInfo.DownloadSate.FAILED)
            View.GONE
        else View.VISIBLE
        progressBar.visibility =
            if (downloadState == DownloadInfo.DownloadSate.DOWNLOADING || downloadState == DownloadInfo.DownloadSate.PAUSED) View.VISIBLE else View.GONE
        progressBar.progress =
            if (downloadState == DownloadInfo.DownloadSate.DOWNLOADING || downloadState == DownloadInfo.DownloadSate.PAUSED) (modelItemDownloadState.downloadInfo!!.progress * 100).toInt() else 0
        when (downloadState) {
            DownloadInfo.DownloadSate.NOT_START -> tvStatus.text =
                tvStatus.resources.getString(R.string.download_not_started,
                    if (modelItemDownloadState.downloadInfo!!.totalSize > 0) {
                        FileUtils.formatFileSize(modelItemDownloadState.downloadInfo!!.totalSize)
                    } else {
                        ""
                    })

            DownloadInfo.DownloadSate.COMPLETED -> tvStatus.text =
                tvStatus.resources.getString(R.string.downloaded_click_to_chat, FileUtils.getFileSizeString(modelDownloadManager.getDownloadedFile(hfModelItem.modelId!!)))

            DownloadInfo.DownloadSate.DOWNLOADING -> if (TextUtils.equals(
                    "Preparing",
                    modelItemDownloadState.downloadInfo?.progressStage
                )
            ) {
                tvStatus.text = tvStatus.resources.getString(R.string.download_preparing)
            } else {
                updateProgress(modelItemDownloadState.downloadInfo!!)
            }

            DownloadInfo.DownloadSate.FAILED -> tvStatus.text = tvStatus.resources.getString(
                R.string.download_failed_click_retry,
                modelItemDownloadState.downloadInfo!!.errorMessage
            )

            DownloadInfo.DownloadSate.PAUSED -> tvStatus.text = tvStatus.resources.getString(
                R.string.downloading_paused,
                if (modelItemDownloadState.downloadInfo!!.totalSize > 0) {
                    FileUtils.formatFileSize(modelItemDownloadState.downloadInfo!!.totalSize)
                } else {
                    ""
                },
                modelItemDownloadState.downloadInfo!!.progress * 100
            )

            else -> {}
        }
    }

    @SuppressLint("DefaultLocale")
    fun updateProgress(downloadInfo: DownloadInfo) {
        progressBar.progress = (downloadInfo.progress * 100).toInt()
        tvStatus.text = itemView.resources.getString(
            R.string.downloading_progress,
            if ((modelItemDownloadState?.downloadInfo?.totalSize ?: 0) > 0) {
                FileUtils.formatFileSize(modelItemDownloadState!!.downloadInfo!!.totalSize)
            } else {
                ""
            },
            downloadInfo.progress * 100,
            downloadInfo.speedInfo
        )
    }

    override fun onClick(v: View) {
        val hfModelItem = v.tag as ModelItem
        modelItemListener.onItemClicked(hfModelItem)
    }

    override fun onLongClick(v: View): Boolean {
        val popupMenu = PopupMenu(v.context, tvStatus)
        val inflater = popupMenu.menuInflater
        inflater.inflate(R.menu.model_item_context_menu, popupMenu.menu)
        popupMenu.setOnMenuItemClickListener { item: MenuItem ->
            val hfModelItem = itemView.tag as ModelItem
            val modelId = hfModelItem.modelId
            if (item.itemId == R.id.menu_delete_model) {
                AlertDialog.Builder(v.context)
                    .setTitle(R.string.confirm_delete_model_title)
                    .setMessage(R.string.confirm_delete_model_message)
                    .setPositiveButton(android.R.string.ok) { _, _ ->
                        MainScope().launch {
                            ModelDownloadManager.getInstance(v.context).deleteModel(modelId!!)
                        }
                    }
                    .setNegativeButton(android.R.string.cancel, null)
                    .show()
            } else if (item.itemId == R.id.menu_pause_download) {
                ModelDownloadManager.getInstance(v.context).pauseDownload(modelId!!)
            } else if (item.itemId == R.id.menu_start_download) {
                ModelDownloadManager.getInstance(v.context).startDownload(modelId!!)
            }
            true
        }
        val modelItemState = this.modelItemDownloadState ?: return true
        val downloadState = modelItemState.downloadInfo!!.downlodaState
        if (downloadState != DownloadInfo.DownloadSate.COMPLETED && downloadState != DownloadInfo.DownloadSate.PAUSED && downloadState != DownloadInfo.DownloadSate.FAILED) {
            popupMenu.menu.findItem(R.id.menu_delete_model).setVisible(false)
        }
        if (downloadState != DownloadInfo.DownloadSate.DOWNLOADING) {
            popupMenu.menu.findItem(R.id.menu_pause_download).setVisible(false)
        }
        if (downloadState != DownloadInfo.DownloadSate.PAUSED && downloadState != DownloadInfo.DownloadSate.NOT_START && downloadState != DownloadInfo.DownloadSate.FAILED
        ) {
            popupMenu.menu.findItem(R.id.menu_start_download).setVisible(false)
        }
        popupMenu.show()
        return true
    }
}
