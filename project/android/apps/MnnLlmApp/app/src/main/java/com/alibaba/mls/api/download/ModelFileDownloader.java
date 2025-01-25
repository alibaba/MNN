// Created by ruoyi.sjd on 2024/12/18.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mls.api.download;
import android.util.Log;

import com.alibaba.mls.api.HfApiException;
import com.alibaba.mls.api.HfFileMetadata;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.RandomAccessFile;
import java.util.concurrent.TimeUnit;

import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;

public class ModelFileDownloader {

    public static final String TAG = "RemoteModelDownloader";

    private final OkHttpClient client;

    public ModelFileDownloader() {
        this.client = new OkHttpClient.Builder()
                .connectTimeout(30, TimeUnit.SECONDS)
                .followRedirects(false)
                .followSslRedirects(false)
                .build();
    }

    public void downloadFile(FileDownloadTask fileDownloadTask, FileDownloadListener fileDownloadListener) throws HfApiException, DownloadPausedException, IOException {
        // Create necessary directories
        fileDownloadTask.pointerPath.getParentFile().mkdirs();
        fileDownloadTask.blobPath.getParentFile().mkdirs();

        if (fileDownloadTask.pointerPath.exists()) {
            Log.d(TAG, "DownloadFile " + fileDownloadTask.relativePath + " already exists");
            return;
        }

        if (fileDownloadTask.blobPath.exists()) {
            HfFileUtils.createSymlink(fileDownloadTask.blobPath.toString(), fileDownloadTask.pointerPath.toString());
            Log.d(TAG, "DownloadFile " + fileDownloadTask.relativePath + " already exists just create symlink");
            return;
        }
        synchronized (this) {
            HfFileMetadata hfFileMetadata = fileDownloadTask.hfFileMetadata;
            downloadToTmpAndMove(fileDownloadTask.blobPathIncomplete,
                    fileDownloadTask.blobPath,
                    hfFileMetadata.location,
                    hfFileMetadata.size,
                    fileDownloadTask.relativePath, false, fileDownloadTask.resumeSize, fileDownloadListener);
            HfFileUtils.createSymlink(fileDownloadTask.blobPath.toPath(), fileDownloadTask.pointerPath.toPath());
        }
    }

    private void downloadToTmpAndMove(File incompletePath, File destinationPath, String urlToDownload,
                                      long expectedSize, String fileName, boolean forceDownload,
                                      long resumeSize,
                                      FileDownloadListener fileDownloadListener) throws HfApiException, DownloadPausedException {
        if (destinationPath.exists() && !forceDownload) {
            return;
        }
        if (incompletePath.exists() && forceDownload) {
            incompletePath.delete();
        }
        downloadChunk(urlToDownload, incompletePath, resumeSize, expectedSize, fileName, fileDownloadListener);
        HfFileUtils.moveWithPermissions(incompletePath, destinationPath);
    }

    private void downloadChunk(String url, File tempFile, long resumeSize, long expectedSize,
                               String displayedFilename, FileDownloadListener fileDownloadListener) throws HfApiException, DownloadPausedException {
        Request.Builder requestBuilder = new Request.Builder()
                .url(url)
                .get()
                .header("Accept-Encoding", "identity");
        long downloadedBytes = resumeSize;

        if (resumeSize > 0) {
            requestBuilder.header("Range", "bytes=" + resumeSize + "-");
        }

        Request request = requestBuilder.build();

        try (Response response = client.newCall(request).execute()) {
            if (response.isSuccessful() || response.code() == 416) {
                try (InputStream is = response.body().byteStream();
                     RandomAccessFile raf = new RandomAccessFile(tempFile, "rw")) {
                    raf.seek(resumeSize);
                    byte[] buffer = new byte[8192];
                    int bytesRead;
                    while ((bytesRead = is.read(buffer)) != -1) {
                        raf.write(buffer, 0, bytesRead);
                        downloadedBytes += bytesRead;
                        if (fileDownloadListener != null) {
                            boolean paused = fileDownloadListener.onDownloadDelta(displayedFilename, downloadedBytes, expectedSize,  bytesRead);
                            if (paused) {
                                throw new DownloadPausedException("Download paused");
                            }
                        }
                    }
                }
            } else {
                throw new HfApiException("HTTP error: " + response.code());
            }
        } catch (IOException e) {
            throw new HfApiException( "Connection error: " + e.getMessage());
        }
    }

    public interface FileDownloadListener {
        boolean onDownloadDelta(String fileName, long downloadedBytes, long totalBytes, long delta);
    }
}
