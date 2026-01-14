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
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import com.alibaba.mls.api.download.DownloadListener
import com.alibaba.mls.api.download.DownloadInfo
import com.alibaba.mls.api.download.DownloadState

class ModelListPresenter(private val context: Context, private val view: ModelListContract.View) :
    ModelItemListener, DownloadListener {
    private val modelListAdapter: ModelListAdapter?
    private var lastClickTime: Long = -1
    private var mainHandler: Handler?
    private val presenterScope = CoroutineScope(Dispatchers.Main + SupervisorJob())

    init {
        this.modelListAdapter = view.adapter
        this.mainHandler = Handler(context.mainLooper)
        ModelDownloadManager.getInstance(context).addListener(this)
        Log.d(TAG, "ModelListPresenter created")
    }

    fun onCreate() {
        Log.d(TAG, "onCreate - starting Flow subscription")
        
        // Subscribe to model list state changes - auto-initialization will happen
        presenterScope.launch {
            ModelListManager.observeModelList().collect { state ->
                handleModelListState(state)
            }
        }
    }

    private fun handleModelListState(state: ModelListManager.ModelListState) {
        when (state) {
            is ModelListManager.ModelListState.Loading -> {
                Log.d(TAG, "State: Loading")
                view.onLoading()
            }
            
            is ModelListManager.ModelListState.Success -> {
                Log.d(TAG, "State: Success (${state.models.size} models, source: ${state.source})")
                
                // Sort models
                val sortedWrappers = state.models.sortedBy { it.hasUpdate }
                
                // Check for updates in background
                presenterScope.launch {
                    try {
                        for (modelWrapper in sortedWrappers) {
                            if (!modelWrapper.modelItem.isLocal) {
                                ModelDownloadManager.getInstance(context)
                                    .checkForUpdate(modelWrapper.modelItem.modelId)
                            }
                        }
                    } catch (e: Exception) {
                        Log.w(TAG, "Error checking for updates", e)
                    }
                }
                
                // Update UI
                onListAvailable(sortedWrappers)
            }
            
            is ModelListManager.ModelListState.Error -> {
                Log.e(TAG, "State: Error", state.exception)
                view.onListLoadError(state.exception.message)
            }
        }
    }

    private fun onListAvailable(modelWrappers: List<ModelItemWrapper>) {
        Log.d(TAG, "onListAvailable: received ${modelWrappers.size} modelWrappers")
        modelListAdapter!!.updateItems(modelWrappers)
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
        // Model deleted, notify manager to refresh
        presenterScope.launch {
            ModelListManager.notifyModelListMayChange(ModelListManager.ChangeReason.MODEL_DELETED)
        }
    }

    override fun onItemUpdate(modelItem: ModelItem) {
        // Start update download for the model
        modelItem.modelId?.let { modelId ->
            Log.d(TAG, "onItemUpdate: starting download for $modelId")
            presenterScope.launch {
                try {
                    ModelDownloadManager.getInstance(context).startDownload(modelId)
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to start download for $modelId", e)
                }
            }
        }
    }

    fun refreshList() {
        Log.d(TAG, "refreshList: triggering model list refresh")
        presenterScope.launch {
            ModelListManager.notifyModelListMayChange(ModelListManager.ChangeReason.MANUAL_REFRESH)
        }
    }

    fun handlePinStateChange(isPinned: Boolean) {
        Log.d(TAG, "handlePinStateChange: isPinned=$isPinned")
        presenterScope.launch {
            val reason = if (isPinned) {
                ModelListManager.ChangeReason.MODEL_PINNED
            } else {
                ModelListManager.ChangeReason.MODEL_UNPINNED
            }
            ModelListManager.notifyModelListMayChange(reason)
        }
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
        Log.d(TAG, "onDownloadStart: $modelId")
        mainHandler?.post {
            // Update UI when model starts updating
            modelListAdapter?.updateItem(modelId)
        }
    }

    override fun onDownloadProgress(modelId: String, downloadInfo: DownloadInfo) {
        Log.d(TAG, "onDownloadProgress: $modelId state=${downloadInfo.downloadState} progress=${downloadInfo.progress}")
        if (downloadInfo.downloadState == DownloadState.COMPLETED) {
            Log.d(TAG, "onDownloadProgress: detected COMPLETED state for $modelId, manually triggering finish")
            onDownloadFinished(modelId, "")
            return
        }
        mainHandler?.post {
            // Update progress for models that are being updated
            modelListAdapter?.updateProgress(modelId, downloadInfo)
        }
    }

    override fun onDownloadFailed(modelId: String, exception: Exception) {
        Log.e(TAG, "onDownloadFailed: $modelId", exception)
        // Not needed for model list
    }

    override fun onDownloadFinished(modelId: String, path: String) {
        Log.d(TAG, "onDownloadFinished: $modelId path=$path")
        // Download completed, notify manager to refresh
        presenterScope.launch {
            ModelListManager.notifyModelListMayChange(ModelListManager.ChangeReason.MODEL_DOWNLOADED)
        }
    }

    override fun onDownloadPaused(modelId: String) {
        // Not needed for model list
    }

    override fun onDownloadFileRemoved(modelId: String) {
        // File removed, notify manager to refresh
        presenterScope.launch {
            ModelListManager.notifyModelListMayChange(ModelListManager.ChangeReason.MODEL_DELETED)
        }
    }

    fun onDestroy() {
        // Cancel coroutine scope to stop Flow collection
        presenterScope.cancel()
        ModelDownloadManager.getInstance(context).removeListener(this)
        mainHandler?.removeCallbacksAndMessages(null)
        this.mainHandler = null
    }

    companion object {
        const val TAG: String = "ModelListPresenter"
    }
}