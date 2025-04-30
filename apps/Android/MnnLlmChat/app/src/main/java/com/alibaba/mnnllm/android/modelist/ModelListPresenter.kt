// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.modelist

import android.content.Context
import android.os.Handler
import android.text.TextUtils
import android.util.Log
import android.widget.Toast
import com.alibaba.mls.api.HfApiClient
import com.alibaba.mls.api.HfApiClient.RepoSearchCallback
import com.alibaba.mls.api.ModelItem
import com.alibaba.mls.api.download.DownloadInfo
import com.alibaba.mls.api.download.DownloadListener
import com.alibaba.mls.api.download.ModelDownloadManager
import com.alibaba.mnnllm.android.R
import com.alibaba.mnnllm.android.utils.ModelUtils.processList
import com.google.gson.Gson
import com.google.gson.GsonBuilder
import com.google.gson.JsonSyntaxException
import com.google.gson.reflect.TypeToken
import java.io.BufferedReader
import java.io.File
import java.io.FileNotFoundException
import java.io.FileReader
import java.io.FileWriter
import java.io.IOException
import java.io.InputStreamReader

class ModelListPresenter(private val context: Context, private val view: ModelListContract.View) :
    ModelItemListener, DownloadListener {
    private var bestApiClient: HfApiClient? = null
    private val modelListAdapter: ModelListAdapter?
    private var lastClickTime: Long = -1

    private val modelItemDownloadStatesMap: MutableMap<String, ModelItemDownloadState> = HashMap()
    private val lastUpdateTimeMap: MutableMap<String, Long> = HashMap()
    private val lastDownloadProgressStage: MutableMap<String, String> = HashMap()

    private var networkErrorCount = 0
    private val modelDownloadManager: ModelDownloadManager =
        ModelDownloadManager.getInstance(this.context)
    private var mainHandler: Handler?

    init {
        modelDownloadManager.setListener(this)
        this.modelListAdapter = view.adapter
        this.mainHandler = Handler(context.mainLooper)
    }

    fun onCreate() {
        modelDownloadManager.setListener(this)
        val items = loadFromCache()
        if (items != null) {
            onListAvailable(items, null)
        }
        requestRepoList(null)
    }

    fun load() {
        requestRepoList(null)
    }

    private fun getModelItemState(hfModelItems: List<ModelItem>?): Map<String, ModelItemDownloadState> {
        modelItemDownloadStatesMap.clear()
        if (hfModelItems == null) {
            return modelItemDownloadStatesMap
        }
        for (repoItem in hfModelItems) {
            val modelItemDownloadState = ModelItemDownloadState()
            modelItemDownloadState.downloadInfo = modelDownloadManager.getDownloadInfo(repoItem.modelId!!)
            modelItemDownloadStatesMap[repoItem.modelId!!] = modelItemDownloadState
        }
        return modelItemDownloadStatesMap
    }

    private fun requestRepoList(onSuccess: Runnable?) {
        view.onLoading()
        networkErrorCount = 0
        if (bestApiClient != null) {
            requestRepoListWithClient(bestApiClient!!, bestApiClient!!.host, 1, onSuccess)
        } else {
            val defaultApiClient = HfApiClient(HfApiClient.HOST_DEFAULT)
            val mirrorApiClient = HfApiClient(HfApiClient.HOST_MIRROR)
            requestRepoListWithClient(defaultApiClient, HfApiClient.HOST_DEFAULT, 2, onSuccess)
            requestRepoListWithClient(mirrorApiClient, HfApiClient.HOST_MIRROR, 2, onSuccess)
        }
    }

    private fun saveToCache(hfModelItems: List<ModelItem>) {
        val gson = GsonBuilder().setPrettyPrinting().create()
        val json = gson.toJson(hfModelItems)

        try {
            FileWriter(context.filesDir.toString() + "/model_list.json").use { writer ->
                writer.write(json)
            }
        } catch (e: IOException) {
            Log.e(TAG, "saveToCacheError")
        }
    }

    private fun loadFromAssets(context: Context, fileName: String): List<ModelItem>? {
        val assetManager = context.assets
        try {
            val inputStream = assetManager.open(fileName)
            val inputStreamReader = InputStreamReader(inputStream)
            val bufferedReader = BufferedReader(inputStreamReader)
            val gson = Gson()
            val listType = object : TypeToken<List<ModelItem?>?>() {}.type
            return gson.fromJson(bufferedReader, listType)
        } catch (e: IOException) {
            Log.e(TAG, "loadFromAssets: Error reading from assets", e)
            return null
        } catch (e: JsonSyntaxException) {
            Log.e(TAG, "loadFromAssets: Invalid JSON in assets", e)
            return null
        }
    }

    private fun loadFromCache(): List<ModelItem>? {
        val cacheFileName = "model_list.json"
        val cacheFile = File(context.filesDir, "model_list.json")
        if (!cacheFile.exists()) {
            return loadFromAssets(this.context, cacheFileName)
        }
        try {
            FileReader(cacheFile.absolutePath).use { reader ->
                val gson = Gson()
                val listType = object : TypeToken<List<ModelItem?>?>() {}.type
                return gson.fromJson(reader, listType)
            }
        } catch (e: FileNotFoundException) {
            Log.d(TAG, "Cache file not found.")
            return null
        } catch (e: IOException) {
            Log.e(TAG, "loadFromCacheError", e) // Log the full exception for debugging
            return null
        } catch (e: JsonSyntaxException) {
            Log.e(TAG, "loadFromCacheError: Invalid JSON", e)
            return null
        }
    }


    private fun onListAvailable(hfModelItems: List<ModelItem>, onSuccess: Runnable?) {
        val hfRepoItemsProcessed = processList(hfModelItems)
        for (item in hfModelItems) {
            modelDownloadManager.getDownloadInfo(item.modelId!!)
        }
        modelListAdapter!!.updateItems(hfRepoItemsProcessed, getModelItemState(hfModelItems))
        onSuccess?.run()
        view.onListAvailable()
    }

    private fun requestRepoListWithClient(
        hfApiClient: HfApiClient,
        tag: String,
        loadCount: Int,
        onSuccess: Runnable?
    ) {
        hfApiClient.searchRepos("", object : RepoSearchCallback {
            override fun onSuccess(hfModelItems: List<ModelItem>) {
                if (bestApiClient == null) {
                    bestApiClient = hfApiClient
                    saveToCache(hfModelItems)
                    onListAvailable(hfModelItems, onSuccess)
                    HfApiClient.bestClient = bestApiClient
                }
                Log.d(
                    TAG,
                    "requestRepoListWithClient success : $tag"
                )
            }

            override fun onFailure(error: String?) {
                networkErrorCount++
                Log.d(
                    TAG,
                    "on requestRepoListWithClient Failure $error tag:$tag"
                )
                if (networkErrorCount == loadCount) {
                    Log.e(
                        TAG,
                        "on requestRepoListWithClient Failure With Retry $error"
                    )
                    view.onListLoadError(error)
                }
            }
        })
    }


    override fun onItemClicked(hfModelItem: ModelItem) {
        //avoid click too fast
        val now = System.currentTimeMillis()
        if (now - this.lastClickTime < 500) {
            return
        }
        this.lastClickTime = now
        if (hfModelItem.isLocal) {
            view.runModel(hfModelItem.localPath!!, hfModelItem.modelId)
            return
        }
        val downloadInfo = modelDownloadManager.getDownloadInfo(hfModelItem.modelId!!)
        if (downloadInfo.downlodaState == DownloadInfo.DownloadSate.COMPLETED) {
            val localDownloadPath = modelDownloadManager.getDownloadedFile(hfModelItem.modelId!!)
            view.runModel(localDownloadPath!!.absolutePath, hfModelItem.modelId)
        } else if (downloadInfo.downlodaState == DownloadInfo.DownloadSate.NOT_START || downloadInfo.downlodaState == DownloadInfo.DownloadSate.FAILED || downloadInfo.downlodaState == DownloadInfo.DownloadSate.PAUSED) {
            modelDownloadManager.startDownload(hfModelItem.modelId!!)
        } else if (downloadInfo.downlodaState == DownloadInfo.DownloadSate.DOWNLOADING) {
            Toast.makeText(
                this.context,
                context.resources.getString(R.string.downloading_please_wait), Toast.LENGTH_SHORT
            ).show()
        }
    }

    override fun onDownloadStart(modelId: String) {
        lastDownloadProgressStage.remove(modelId)
        lastUpdateTimeMap.remove(modelId)
    }

    override fun onDownloadFailed(modelId: String, hfApiException: Exception) {
        if (this.mainHandler != null) {
            mainHandler!!.post {
                modelListAdapter!!.updateItem(
                    modelId
                )
            }
        }
    }

    override fun onDownloadProgress(modelId: String, progress: DownloadInfo) {
        val lastUpdateTime = lastUpdateTimeMap[modelId]
        val now = System.currentTimeMillis()
        val progressStateChanged = !TextUtils.equals(
            progress.progressStage,
            lastDownloadProgressStage[modelId]
        )
        if (!progressStateChanged && lastUpdateTime != null && now - lastUpdateTime < 500) {
            return
        }
        lastDownloadProgressStage[modelId] = progress.progressStage!!
        lastUpdateTimeMap[modelId] = now
        if (lastUpdateTime != null && lastUpdateTime > 0 && !progressStateChanged) {
            if (this.mainHandler != null) {
                mainHandler!!.post {
                    modelListAdapter!!.updateProgress(
                        modelId,
                        progress
                    )
                }
            }
        } else {
            if (this.mainHandler != null) {
                mainHandler!!.post {
                    modelListAdapter!!.updateItem(
                        modelId
                    )
                }
            }
        }
    }

    override fun onDownloadFinished(modelId: String, path: String) {
        if (this.mainHandler != null) {
            mainHandler!!.post {
                modelListAdapter!!.updateItem(
                    modelId
                )
            }
        }
    }

    override fun onDownloadPaused(modelId: String) {
        if (this.mainHandler != null) {
            mainHandler!!.post {
                modelListAdapter!!.updateItem(
                    modelId
                )
            }
        }
    }

    override fun onDownloadFileRemoved(modelId: String) {
        mainHandler!!.post {
            modelListAdapter!!.updateItem(
                modelId
            )
        }
    }

    val unfinishedDownloadCount: Int
        get() = modelDownloadManager.unfinishedDownloadsSize

    fun resumeAllDownloads() {
        modelDownloadManager.resumeAllDownloads()
    }

    fun onDestroy() {
        mainHandler!!.removeCallbacksAndMessages(null)
        this.mainHandler = null
    }

    companion object {
        const val TAG: String = "ModelListPresenter"
    }
}
