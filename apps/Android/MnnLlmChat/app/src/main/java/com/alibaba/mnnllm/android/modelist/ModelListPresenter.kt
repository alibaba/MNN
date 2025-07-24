// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.modelist

import android.content.Context
import android.os.Handler
import android.util.Log
import com.alibaba.mls.api.ModelItem
import com.alibaba.mls.api.download.ModelDownloadManager
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import com.alibaba.mls.api.download.DownloadListener
import com.alibaba.mls.api.download.DownloadInfo

class ModelListPresenter(private val context: Context, private val view: ModelListContract.View) :
    ModelItemListener, DownloadListener {
    private val modelListAdapter: ModelListAdapter?
    private var lastClickTime: Long = -1
    private var mainHandler: Handler?
    private val presenterScope = CoroutineScope(Dispatchers.Main)
    private var loading = false

    init {
        this.modelListAdapter = view.adapter
        this.mainHandler = Handler(context.mainLooper)
        ModelDownloadManager.getInstance(context).addListener(this)
        Log.d(TAG, "onCreate: ModelListPresenter created", Throwable())
    }

    fun onCreate() {
        loadDownloadedModels()
    }

    fun load() {
        loadDownloadedModels()
    }

    private fun loadDownloadedModels() {
        if (loading) {
            return
        }
        loading = true
        view.onLoading()
        presenterScope.launch {
            try {
                val modelWrappers = ModelListManager.loadAvailableModels(context).toMutableList()
                val sortedWrappers = modelWrappers.sortedBy { it.hasUpdate }
                launch {
                    for (modelWrapper in sortedWrappers) {
                        if (!modelWrapper.modelItem.isLocal) {
                            ModelDownloadManager.getInstance(context).checkForUpdate(modelWrapper.modelItem.modelId)
                        }
                    }
                }
                onListAvailable(sortedWrappers, null)
            } catch (e: Exception) {
                Log.e(TAG, "Error loading downloaded models", e)
                view.onListLoadError(e.message)
            }
        }
        loading = false
    }

    private fun onListAvailable(modelWrappers: List<ModelItemWrapper>, onSuccess: Runnable?) {
        Log.d(TAG, "onListAvailable: received ${modelWrappers.size} modelWrappers")
        modelListAdapter!!.updateItems(modelWrappers)
        onSuccess?.run()
        Log.d(TAG, "onListAvailable: calling view.onListAvailable()")
        view.onListAvailable()
    }


    override fun onItemClicked(modelItem: ModelItem) {
        val now = System.currentTimeMillis()
        if (now - this.lastClickTime < 500) {
            return
        }
        this.lastClickTime = now
        if (modelItem.isLocal) {
            view.runModel(null, modelItem.modelId)
            return
        }
        view.runModel(null, modelItem.modelId)
    }

    override fun onItemLongClicked(modelItem: ModelItem): Boolean {
        // This is handled by the adapter's wrapper listener, not here
        return false
    }

    override fun onItemDeleted(modelItem: ModelItem) {
        // Refresh the list after model deletion
        refreshList()
    }

    override fun onItemUpdate(modelItem: ModelItem) {
        // Start update download for the model
        modelItem.modelId?.let { modelId ->
            ModelDownloadManager.getInstance(context).startDownload(modelId)
        }
    }

    fun refreshList() {
        loadDownloadedModels()
    }

    // DownloadListener implementation
    override fun onDownloadTotalSize(modelId: String, totalSize: Long) {
        // Not needed for model list
    }

    override fun onDownloadHasUpdate(modelId: String, downloadInfo: DownloadInfo) {
        Log.d(TAG, "onDownloadHasUpdate: $modelId")
        mainHandler?.post {
            // Update the hasUpdate flag and refresh the specific item
            modelListAdapter?.updateItemHasUpdate(modelId, true)
        }
    }

    override fun onDownloadStart(modelId: String) {
        mainHandler?.post {
            // Update UI when model starts updating
            modelListAdapter?.updateItem(modelId)
        }
    }

    override fun onDownloadProgress(modelId: String, downloadInfo: DownloadInfo) {
        mainHandler?.post {
            // Update progress for models that are being updated
            modelListAdapter?.updateProgress(modelId, downloadInfo)
        }
    }

    override fun onDownloadFailed(modelId: String, exception: Exception) {
        // Not needed for model list
    }

    override fun onDownloadFinished(modelId: String, path: String) {
        mainHandler?.post {
            // Refresh the list after download completion
            refreshList()
        }
    }

    override fun onDownloadPaused(modelId: String) {
        // Not needed for model list
    }

    override fun onDownloadFileRemoved(modelId: String) {
        mainHandler?.post {
            // Refresh the list after model removal
            refreshList()
        }
    }

    fun onDestroy() {
        ModelDownloadManager.getInstance(context).removeListener(this)
        mainHandler!!.removeCallbacksAndMessages(null)
        this.mainHandler = null
    }

    companion object {
        const val TAG: String = "ModelListPresenter"
    }
}
