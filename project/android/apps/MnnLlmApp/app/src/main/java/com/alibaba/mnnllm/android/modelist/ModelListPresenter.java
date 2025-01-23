// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.modelist;

import static com.alibaba.mls.api.HfApiClient.HOST_DEFAULT;
import static com.alibaba.mls.api.HfApiClient.HOST_MIRROR;

import android.content.Context;
import android.os.Handler;
import android.text.TextUtils;
import android.util.Log;
import android.widget.Toast;

import com.alibaba.mls.api.HfApiClient;
import com.alibaba.mls.api.HfRepoItem;
import com.alibaba.mls.api.download.DownloadInfo;
import com.alibaba.mls.api.download.DownloadListener;
import com.alibaba.mls.api.download.ModelDownloadManager;
import com.alibaba.mnnllm.android.utils.ModelUtils;
import com.alibaba.mnnllm.android.R;

import java.io.File;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ModelListPresenter implements ModelItemListener, DownloadListener {

    public static final String TAG = "ModelListPresenter";
    private HfApiClient bestApiClient;
    private ModelListAdapter modelListAdapter;
    private Context context;
    private ModelListContract.View view;
    private long lastClickTime = -1;

    private Map<String, ModelItemState> modelItemStatesMap = new HashMap<>();
    private Map<String, Long> lastUpdateTimeMap = new HashMap<>();
    private Map<String, String> lastDownloadProgressStage = new HashMap<>();

    private int networkErrorCount = 0;
    private ModelDownloadManager modelDownloadManager;
    private Handler mainHandler;

    public ModelListPresenter(Context context, ModelListContract.View view) {
        this.context = context;
        this.view = view;
        modelDownloadManager = ModelDownloadManager.getInstance(this.context);
        modelDownloadManager.setListener(this);
        this.modelListAdapter = view.getAdapter();
        this.mainHandler = new Handler(context.getMainLooper());
    }

    public void onCreate() {
        modelDownloadManager.setListener(this);
        requestRepoList(null);
    }

    void load() {
        requestRepoList(null);
    }

    Map<String, ModelItemState> getModelItemState(List<HfRepoItem> hfRepoItems) {
        modelItemStatesMap.clear();
        if (hfRepoItems == null) {
            return modelItemStatesMap;
        }
        for (HfRepoItem repoItem : hfRepoItems) {
            ModelItemState modelItemState = new ModelItemState();
            modelItemState.downloadInfo = modelDownloadManager.getDownloadInfo(repoItem.getModelId());
            modelItemStatesMap.put(repoItem.getModelId(), modelItemState);
        }
        return modelItemStatesMap;
    }

    private void requestRepoList(Runnable onSuccess) {
        this.view.onLoading();
        networkErrorCount = 0;
        if (bestApiClient != null) {
            requestRepoListWithClient(bestApiClient, bestApiClient.getHost(), 1, onSuccess);
        } else {
            HfApiClient defaultApiClient = new HfApiClient(HOST_DEFAULT);
            HfApiClient mirrorApiClient = new HfApiClient(HOST_MIRROR);
            requestRepoListWithClient(defaultApiClient, HOST_DEFAULT, 2, onSuccess);
            requestRepoListWithClient(mirrorApiClient, HOST_MIRROR, 2, onSuccess);
        }
    }

    private void requestRepoListWithClient(HfApiClient hfApiClient, String tag, int loadCount, Runnable onSuccess) {
        hfApiClient.searchRepos("", new HfApiClient.RepoSearchCallback() {
            @Override
            public void onSuccess(List<HfRepoItem> hfRepoItems) {
                if (bestApiClient == null) {
                    bestApiClient =  hfApiClient;
                    if (hfRepoItems != null) {
                        hfRepoItems = ModelUtils.processList(hfRepoItems);
                        for (HfRepoItem item : hfRepoItems) {
                            modelDownloadManager.getDownloadInfo(item.getModelId());
                        }
                        modelListAdapter.updateItems(hfRepoItems, getModelItemState(hfRepoItems));
                        if (onSuccess != null) {
                            onSuccess.run();
                        }
                    }
                    view.onListAvailable();
                    HfApiClient.setBestClient(bestApiClient);
                }
                Log.d(TAG, "requestRepoListWithClient success : " + tag);
            }

            @Override
            public void onFailure(String error) {
                networkErrorCount++;
                Log.d(TAG, "on requestRepoListWithClient Failure " +  error + " tag:" + tag);
                if (networkErrorCount == loadCount) {
                    Log.e(TAG, "on requestRepoListWithClient Failure With Retry " +  error);
                    view.onListLoadError(error);
                }
            }
        });
    }


    @Override
    public void onItemClicked(HfRepoItem hfRepoItem) {
        //avoid click too fast
        long now = System.currentTimeMillis();
        if (now - this.lastClickTime < 500) {
            return;
        }
        this.lastClickTime = now;
        DownloadInfo downloadInfo = modelDownloadManager.getDownloadInfo(hfRepoItem.getModelId());
        if (downloadInfo.downlodaState == DownloadInfo.DownloadSate.COMPLETED) {
            File localDownloadPath = modelDownloadManager.getDownloadPath(hfRepoItem.getModelId());
            this.view.runModel(localDownloadPath.getAbsolutePath(), hfRepoItem.getModelName());
        } else if (downloadInfo.downlodaState == DownloadInfo.DownloadSate.NOT_START
                || downloadInfo.downlodaState == DownloadInfo.DownloadSate.FAILED ||
                 downloadInfo.downlodaState == DownloadInfo.DownloadSate.PAUSED) {
            modelDownloadManager.startDownload(hfRepoItem.getModelId());
        } else if (downloadInfo.downlodaState == DownloadInfo.DownloadSate.DOWNLOADING){
            Toast.makeText(this.context, this.context.getResources().getString(R.string.downloading_please_wait), Toast.LENGTH_SHORT).show();
        }
    }

    @Override
    public void onDownloadStart(String modelId) {
        lastDownloadProgressStage.remove(modelId);
        lastUpdateTimeMap.remove(modelId);
    }

    @Override
    public void onDownloadFailed(String modelId, Exception hfApiException) {
        this.mainHandler.post(() -> {
            this.modelListAdapter.updateItem(modelId);
        });
    }

    @Override
    public void onDownloadProgress(String modelId, DownloadInfo downloadInfo) {
        Long lastUpdateTime = lastUpdateTimeMap.get(modelId);
        Long now = System.currentTimeMillis();
        boolean progressStateChanged = !TextUtils.equals(downloadInfo.progressStage, lastDownloadProgressStage.get(modelId));
        if (!progressStateChanged && lastUpdateTime != null && now - lastUpdateTime < 500) {
            return;
        }
        lastDownloadProgressStage.put(modelId, downloadInfo.progressStage);
        lastUpdateTimeMap.put(modelId, now);
        if (lastUpdateTime != null && lastUpdateTime > 0 && !progressStateChanged) {
            this.mainHandler.post(() -> {
                this.modelListAdapter.updateProgres(modelId, downloadInfo.progress);
            });
        } else {
            this.mainHandler.post(() -> {
                this.modelListAdapter.updateItem(modelId);
            });
        }
    }

    @Override
    public void onDownloadFinished(String modelId, String path) {
        this.mainHandler.post(() -> {
            this.modelListAdapter.updateItem(modelId);
        });
    }

    @Override
    public void onDownloadPaused(String modelId) {
        this.mainHandler.post(() -> {
            this.modelListAdapter.updateItem(modelId);
        });
    }

    @Override
    public void onDownloadFileRemoved(String modelId) {
        this.mainHandler.post(() -> {
            this.modelListAdapter.updateItem(modelId);
        });
    }

    public int getUnfisnishedDownloadsSize() {
        return modelDownloadManager.getUnfinishedDownloadsSize();
    }

    public void resumeAllDownloads() {
        modelDownloadManager.resumeAllDownloads();
    }

    public void onDestroy() {
        this.mainHandler.removeCallbacksAndMessages(null);
        this.mainHandler = null;
    }
}
