// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mls.api.download;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Build;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.alibaba.mls.api.ApplicationUtils;
import com.alibaba.mls.api.HfApiClient;
import com.alibaba.mls.api.HfApiException;
import com.alibaba.mls.api.HfFileMetadata;
import com.alibaba.mls.api.HfRepoInfo;
import java.util.concurrent.atomic.AtomicInteger;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.TimeUnit;

import okhttp3.OkHttpClient;

public class ModelDownloadManager {

    private static volatile ModelDownloadManager instance;
    private final Context context;
    private DownloadListener downloadListener;
    private final String cachePath;
    public static final String TAG = "ModelDownloadManager";

    private HfApiClient hfApiClient;
    private OkHttpClient metaInfoClient;
    private final HashMap<String, DownloadInfo> downloadInfoMap = new HashMap<>();
    private final Set<String> pausedSet = Collections.synchronizedSet(new HashSet<>());
    private final Intent foregroundSerivceIntent;

    private final AtomicInteger activeDownloadCount;

    public static final int REQUEST_CODE_POST_NOTIFICATIONS = 998;

    private boolean foregroundServiceStarted = false;

    private ModelDownloadManager(Context context) {
        this.context = context;
        this.cachePath = context.getFilesDir().getAbsolutePath() + "/.mnnmodels";
        foregroundSerivceIntent = new Intent(context.getApplicationContext(), DownlodForegroundService.class);
        this.activeDownloadCount = new AtomicInteger(0);
    }

    public static ModelDownloadManager getInstance(Context context) {
        if (instance == null) {
            synchronized (ModelDownloadManager.class) {
                if (instance == null) {
                    instance = new ModelDownloadManager(context);
                }
            }
        }
        return instance;
    }

    public void setListener(DownloadListener downloadListener) {
        this.downloadListener = downloadListener;
    }

    public File getDownloadPath(String modelId) {
        return new File(cachePath, HfFileUtils.getLastFileName(modelId));
    }

    public void pauseDownload(String modelId) {
        if (getDownloadInfo(modelId).downlodaState != DownloadInfo.DownloadSate.DOWNLOADING) {
            return;
        }
        pausedSet.add(modelId);
    }

    private HfApiClient getApiClient() {
        if (hfApiClient == null) {
            hfApiClient = HfApiClient.getBestClient();
        }
        if (hfApiClient == null) {
            hfApiClient = new HfApiClient(HfApiClient.HOST_DEFAULT);
        }
        return hfApiClient;
    }

    public void startDownload(String modelId) {
        if (downloadListener != null) {
            downloadListener.onDownloadStart(modelId);
        }
        this.updateDownloadingProgress(modelId, "Preparing", null, 0, 10);
        getApiClient().getRepoInfo(modelId, "main", new HfApiClient.RepoInfoCallback() {
            @Override
            public void onSuccess(HfRepoInfo hfRepoInfo) {
                downloadRepo(hfRepoInfo);
            }

            @Override
            public void onFailure(String error) {
                setDownloadFailed(modelId, new HfApiException("getRepoInfoFailed" + error));
            }
        });
    }

    @NonNull
    public DownloadInfo getDownloadInfo(String modelId) {
        if (!downloadInfoMap.containsKey(modelId)) {
            DownloadInfo downloadInfo = new DownloadInfo();
            File downloadPath = getDownloadPath(modelId);
            if ( downloadPath.exists()) {
                downloadInfo.downlodaState = DownloadInfo.DownloadSate.COMPLETED;
                downloadInfo.progress = 1.0;
            } else if (DownloadPersistentData.getDownloadSizeTotal(ApplicationUtils.get(), modelId) > 0) {
                long totalSize = DownloadPersistentData.getDownloadSizeTotal(ApplicationUtils.get(), modelId);
                long savedSize = DownloadPersistentData.getDownloadSizeSaved(ApplicationUtils.get(), modelId);
                downloadInfo.totalSize = totalSize;
                downloadInfo.savedSize = savedSize;
                downloadInfo.progress = (double)savedSize / totalSize;
                downloadInfo.downlodaState = DownloadInfo.DownloadSate.PAUSED;
            } else {
                downloadInfo.downlodaState = DownloadInfo.DownloadSate.NOT_START;
                downloadInfo.progress = 0.0;
            }
            downloadInfoMap.put(modelId, downloadInfo);
        }
        return Objects.requireNonNull(downloadInfoMap.get(modelId));
    }

    private void setDownloadFinished(String modelId, String path) {
        DownloadInfo downloadInfo = getDownloadInfo(modelId);
        downloadInfo.downlodaState = DownloadInfo.DownloadSate.COMPLETED;
        if (downloadListener != null) {
            downloadListener.onDownloadFinished(modelId, path);
        }
    }
    private void setDownloadPaused(String modelId) {
        DownloadInfo downloadInfo = getDownloadInfo(modelId);
        downloadInfo.downlodaState = DownloadInfo.DownloadSate.PAUSED;
        if (downloadListener != null) {
            downloadListener.onDownloadPaused(modelId);
        }
    }

    private void onDownloadTaskAdded(int count) {
        if (count == 1) {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                if (ContextCompat.checkSelfPermission(context, android.Manifest.permission.POST_NOTIFICATIONS)
                        != PackageManager.PERMISSION_GRANTED) {
                    ActivityCompat.requestPermissions(
                            (Activity)context,
                            new String[]{android.Manifest.permission.POST_NOTIFICATIONS},
                            REQUEST_CODE_POST_NOTIFICATIONS
                    );
                } else {
                    ApplicationUtils.get().startForegroundService(foregroundSerivceIntent);
                    foregroundServiceStarted = true;
                }
            }
        }
    }

    private void onDownloadTaskRemoved(int count) {
        if (count == 0) {
            ApplicationUtils.get().stopService(foregroundSerivceIntent);
            foregroundServiceStarted = false;
        }
    }

    private void setDownloadFailed(String modelId, Exception e) {
        Log.e(TAG, "onDownloadFailed: " + modelId, e);
        DownloadInfo downloadInfo = getDownloadInfo(modelId);
        downloadInfo.downlodaState = DownloadInfo.DownloadSate.FAILED;
        downloadInfo.errorMessage = e.getMessage();
        if (downloadListener != null) {
            downloadListener.onDownloadFailed(modelId, e);
        }
    }

    private void updateDownloadingProgress(String modelId, String stage, String currentFile, long saved, long total) {
        if (!downloadInfoMap.containsKey(modelId)) {
            DownloadInfo downloadInfo = new DownloadInfo();
            downloadInfoMap.put(modelId, downloadInfo);
        }
        DownloadInfo downloadInfo = downloadInfoMap.get(modelId);
        assert downloadInfo != null;
        downloadInfo.progress = (double)saved / total;
        downloadInfo.savedSize  = saved;
        downloadInfo.totalSize = total;
        downloadInfo.progressStage = stage;
        downloadInfo.currentFile = currentFile;
        downloadInfo.downlodaState = DownloadInfo.DownloadSate.DOWNLOADING;
        DownloadPersistentData.saveDownloadSizeTotal(ApplicationUtils.get(), modelId, total);
        DownloadPersistentData.saveDownloadSizeSaved(ApplicationUtils.get(), modelId, saved);
        if (downloadListener != null) {
            downloadListener.onDownloadProgress(modelId, downloadInfo);
        }
    }
    private OkHttpClient getMetaInfoHttpClient() {
        if (metaInfoClient == null) {
            metaInfoClient =  new OkHttpClient.Builder()
                    .connectTimeout(30,TimeUnit.SECONDS)
                    .followRedirects(false)
                    .followSslRedirects(false)
                    .build();
        }
        return metaInfoClient;
    }

    private List<HfFileMetadata> requestMedataDataList(HfRepoInfo hfRepoInfo) throws HfApiException {
        List<HfFileMetadata> list = new ArrayList<>();
        for (HfRepoInfo.SiblingItem subFile : hfRepoInfo.getSiblings()) {
            String url = "https://" + this.hfApiClient.getHost() + "/" + hfRepoInfo.getModelId() + "/resolve/main/" + subFile.rfilename;
            HfFileMetadata metaData = HfFileMetadataUtils.getFileMetadata(getMetaInfoHttpClient(), url);
            list.add(metaData);
        }
        return list;
    }

    private List<FileDownloadTask> collectTaskList(File storageFolder, File parentPointerPath, HfRepoInfo hfRepoInfo, long[] totalAndDownloadSize) throws HfApiException {
        HfFileMetadata metaData;
        List<HfFileMetadata> metaDataList = DownloadPersistentData.getMetaData(ApplicationUtils.get(), hfRepoInfo.getModelId());
        Log.d(TAG, "collectTaskList savedMetaDataList: " +  (metaDataList == null ? "null" : metaDataList.size()));
        List<FileDownloadTask> fileDownloadTasks = new ArrayList<>();
        metaDataList = requestMedataDataList(hfRepoInfo);
        DownloadPersistentData.saveMetaData(ApplicationUtils.get(), hfRepoInfo.getModelId(), metaDataList);
        for (int i = 0; i < hfRepoInfo.getSiblings().size(); i++) {
            HfRepoInfo.SiblingItem subFile = hfRepoInfo.getSiblings().get(i);
            metaData = metaDataList.get(i);
            FileDownloadTask fileDownloadTask = new FileDownloadTask();
            fileDownloadTask.relativePath = subFile.rfilename;
            fileDownloadTask.hfFileMetadata = metaData;
            fileDownloadTask.blobPath = new File(storageFolder, "blobs/" + metaData.etag);
            fileDownloadTask.blobPathIncomplete = new File(storageFolder, "blobs/" + metaData.etag + ".incomplete");
            fileDownloadTask.pointerPath = new File(parentPointerPath, subFile.rfilename);
            fileDownloadTask.resumeSize = fileDownloadTask.blobPath.exists() ? fileDownloadTask.blobPath.length() :
                    (fileDownloadTask.blobPathIncomplete.exists() ? fileDownloadTask.blobPathIncomplete.length() : 0);
            totalAndDownloadSize[0] += metaData.size;
            totalAndDownloadSize[1] += fileDownloadTask.resumeSize;
            fileDownloadTasks.add(fileDownloadTask);
        }
        return fileDownloadTasks;
    }
    public void downloadRepo(HfRepoInfo hfRepoInfo) {
        Log.d(TAG, "DownloadStart " + hfRepoInfo.getModelId() + " host: " + getApiClient().getHost());
        DownloadExecutor.getExecutor().submit(() -> {
            onDownloadTaskAdded(this.activeDownloadCount.incrementAndGet());
            downloadRepoInner(hfRepoInfo);
            onDownloadTaskRemoved(this.activeDownloadCount.decrementAndGet());
        });
    }



    private void downloadRepoInner(HfRepoInfo hfRepoInfo) {
        File folderLinkFile = new File(cachePath, HfFileUtils.getLastFileName(hfRepoInfo.getModelId()));
        if (folderLinkFile.exists()) {
            setDownloadFinished(hfRepoInfo.getModelId(), folderLinkFile.getAbsolutePath());
            return;
        }
        ModelFileDownloader modelDownloader = new ModelFileDownloader();
        Log.d(TAG, "Repo SHA: " + hfRepoInfo.getSha());

        boolean hasError = false;
        StringBuilder errorInfo = new StringBuilder();

        String repoFolderName = HfFileUtils.repoFolderName(hfRepoInfo.getModelId(), "model");
        File storageFolder = new File(cachePath, repoFolderName);
        File parentPointerPath = HfFileUtils.getPointerPathParent(storageFolder, hfRepoInfo.getSha());
        List<FileDownloadTask> downloadTaskList;
        long[] totalAndDownloadSize = new long[2];
        try {
            downloadTaskList = collectTaskList(storageFolder, parentPointerPath, hfRepoInfo, totalAndDownloadSize);
        } catch (HfApiException e) {
            setDownloadFailed(hfRepoInfo.getModelId(), e);
            return;
        }
        ModelFileDownloader.FileDownloadListener fileDownloadListener = (filename, downloadedBytes, totalBytes, delta) -> {
            totalAndDownloadSize[1] += delta;
            updateDownloadingProgress(hfRepoInfo.getModelId(), "file", filename,  totalAndDownloadSize[1] , totalAndDownloadSize[0]);
            return pausedSet.contains(hfRepoInfo.getModelId());
        };
        try {
            for (FileDownloadTask fileDownloadTask: downloadTaskList) {
                modelDownloader.downloadFile(fileDownloadTask, fileDownloadListener);
            }
        } catch (DownloadPausedException e) {
            pausedSet.remove(hfRepoInfo.getModelId());
            setDownloadPaused(hfRepoInfo.getModelId());
            return;
        } catch (Exception e) {
            setDownloadFailed(hfRepoInfo.getModelId(), e);
            return;
        }
        if (!hasError) {
            String folderLinkPath = folderLinkFile.getAbsolutePath();
            HfFileUtils.createSymlink(parentPointerPath.toString(), folderLinkPath);
            setDownloadFinished(hfRepoInfo.getModelId(), folderLinkPath);
        } else {
            Log.e(TAG, "Errors occurred during download: " + errorInfo.toString());
        }
    }

    public void removeDownload(String modelId) {
        String repoFolderName = HfFileUtils.repoFolderName(modelId, "model");
        File storageFolder = new File(cachePath, repoFolderName);
        Log.d(TAG, "removeStorageFolder: " + storageFolder.getAbsolutePath());
        if (storageFolder.exists()) {
            boolean result = HfFileUtils.deleteDirectoryRecursively2(storageFolder);
            if (!result) {
                Log.e(TAG, "remove storageFolder" + storageFolder.getAbsolutePath() + " faield");
            }
        }
        DownloadPersistentData.removeProgress(ApplicationUtils.get(), modelId);
        File linkFolder = this.getDownloadPath(modelId);
        Log.d(TAG, "removeLinkFolder: " + linkFolder.getAbsolutePath());
        linkFolder.delete();
        if (downloadListener != null) {
            DownloadInfo downloadInfo = getDownloadInfo(modelId);
            downloadInfo.downlodaState = DownloadInfo.DownloadSate.NOT_START;
            downloadListener.onDownloadFileRemoved(modelId);
        }
    }

    public int getUnfinishedDownloadsSize() {
        int count = 0;
        for (String key: this.downloadInfoMap.keySet()) {
            DownloadInfo downloadInfo = this.downloadInfoMap.get(key);
            assert downloadInfo != null;
            if (downloadInfo.downlodaState == DownloadInfo.DownloadSate.FAILED || downloadInfo.downlodaState == DownloadInfo.DownloadSate.PAUSED) {
                count++;
            }
        }
        return count;
    }

    public void resumeAllDownloads() {
        for (String key: this.downloadInfoMap.keySet()) {
            DownloadInfo downloadInfo = this.downloadInfoMap.get(key);
            if (downloadInfo.downlodaState == DownloadInfo.DownloadSate.FAILED || downloadInfo.downlodaState == DownloadInfo.DownloadSate.PAUSED) {
                startDownload(key);
            }
        }
    }

    public void pauseAllDownloads() {
        for (String key: this.downloadInfoMap.keySet()) {
            DownloadInfo downloadInfo = this.downloadInfoMap.get(key);
            if (downloadInfo.downlodaState == DownloadInfo.DownloadSate.DOWNLOADING) {
                pauseDownload(key);
            }
        }
    }

    public void startForegroundService() {
        if (!foregroundServiceStarted && activeDownloadCount.get() > 0) {
            ApplicationUtils.get().startForegroundService(foregroundSerivceIntent);
        }
    }
}
