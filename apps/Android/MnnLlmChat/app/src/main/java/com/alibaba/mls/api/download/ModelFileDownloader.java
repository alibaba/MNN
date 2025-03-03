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
//                .addInterceptor(OkHttpUtils.createLoggingInterceptor())
                .build();
    }

    public void downloadFile(FileDownloadTask fileDownloadTask, FileDownloadListener fileDownloadListener) throws HfApiException, DownloadPausedException, IOException {
        // Create necessary directories
        Log.d(TAG, "downloadFile inner");
        fileDownloadTask.pointerPath.getParentFile().mkdirs();
        fileDownloadTask.blobPath.getParentFile().mkdirs();

        if (fileDownloadTask.pointerPath.exists()) {
            Log.d(TAG, "DownloadFile " + fileDownloadTask.relativePath + " already exists");
            return;
        }

        if (fileDownloadTask.blobPath.exists()) {
            DownloadFileUtils.createSymlink(fileDownloadTask.blobPath.toString(), fileDownloadTask.pointerPath.toString());
            Log.d(TAG, "DownloadFile " + fileDownloadTask.relativePath + " already exists just create symlink");
            return;
        }
        synchronized (this) {
            HfFileMetadata hfFileMetadata = fileDownloadTask.hfFileMetadata;
            downloadToTmpAndMove(fileDownloadTask,
                    fileDownloadTask.blobPathIncomplete,
                    fileDownloadTask.blobPath,
                    hfFileMetadata.location,
                    hfFileMetadata.size,
                    fileDownloadTask.relativePath, false, fileDownloadListener);
            DownloadFileUtils.createSymlink(fileDownloadTask.blobPath.toPath(), fileDownloadTask.pointerPath.toPath());
        }
    }

    private void downloadToTmpAndMove(FileDownloadTask fileDownloadTask,
                                      File incompletePath, File destinationPath, String urlToDownload,
                                      long expectedSize, String fileName, boolean forceDownload,
                                      FileDownloadListener fileDownloadListener) throws HfApiException, DownloadPausedException {
        if (destinationPath.exists() && !forceDownload) {
            return;
        }
        if (incompletePath.exists() && forceDownload) {
            incompletePath.delete();
        }

        if (fileDownloadTask.downloadedSize >= expectedSize) {
            return;
        }
        Request.Builder requestBuilder = new Request.Builder()
                .url(urlToDownload)
                .get();
        Request request = requestBuilder.build();
        try (Response response = client.newCall(request).execute()) {
            Log.d(TAG, "response code: " + response.code());
            for (String header : response.headers().names()) {
                Log.d(TAG, "downloadToTmpAndMove response header: " + header + ": " + response.header(header));
            }
            if (response.code() == 302 || response.code() == 303) {
                urlToDownload = response.header("Location");
            }
        } catch (IOException e) {
            throw new HfApiException("get header error" + e.getMessage());
        }
        Log.d(TAG, "downloadToTmpAndMove urlToDownload: " + urlToDownload + " to file: " + incompletePath + " to destination: " + destinationPath);
        int maxRetry = 10;
        if (fileDownloadTask.downloadedSize < expectedSize) {
            for (int i = 0; i < maxRetry; i++) {
                try {
                    Log.d(TAG, "downloadChunk try the "  + i + " turn");
                    downloadChunk(fileDownloadTask, urlToDownload, incompletePath, expectedSize, fileName, fileDownloadListener);
                    Log.d(TAG, "downloadChunk try the "  + i + " turn finish");
                    break;
                } catch (DownloadPausedException e) {
                    throw e;
                } catch (Exception e) {
                    if (i == maxRetry -1) {
                        throw e;
                    } else {
                        Log.e(TAG, "downloadChunk failed sleep and retrying: " + e.getMessage());
                        try {
                            Thread.sleep(1000);
                        } catch (InterruptedException ex) {
                            throw new RuntimeException(ex);
                        }
                    }
                }
            }
        }
        DownloadFileUtils.moveWithPermissions(incompletePath, destinationPath);
    }

    private void downloadChunk(FileDownloadTask fileDownloadTask, String url, File tempFile, long expectedSize,
                               String displayedFilename, FileDownloadListener fileDownloadListener) throws HfApiException, DownloadPausedException {
        Request.Builder requestBuilder = new Request.Builder()
                .url(url)
                .get()
                .header("Accept-Encoding", "identity");
        if (fileDownloadTask.downloadedSize >= expectedSize) {
            return;
        }
        long downloadedBytes = fileDownloadTask.downloadedSize;

        if (fileDownloadTask.downloadedSize > 0) {
            requestBuilder.header("Range", "bytes=" + fileDownloadTask.downloadedSize + "-");
        }
        Log.d(TAG, "resume size: " + fileDownloadTask.downloadedSize + " expectedSize: " + expectedSize);
        Request request = requestBuilder.build();
        try (Response response = client.newCall(request).execute()) {
            Log.d(TAG, "downloadChunk response: success: " + response.isSuccessful() + " code: " + response.code());
            if (response.isSuccessful() || response.code() == 416) {
                try (InputStream is = response.body().byteStream();
                     RandomAccessFile raf = new RandomAccessFile(tempFile, "rw")) {
                    raf.seek(fileDownloadTask.downloadedSize);
                    byte[] buffer = new byte[8192];
                    int bytesRead;
                    while ((bytesRead = is.read(buffer)) != -1) {
                        raf.write(buffer, 0, bytesRead);
                        downloadedBytes += bytesRead;
                        fileDownloadTask.downloadedSize += bytesRead;
                        if (fileDownloadListener != null) {
                            boolean paused = fileDownloadListener.onDownloadDelta(displayedFilename, downloadedBytes, expectedSize,  bytesRead);
                            if (paused) {
                                throw new DownloadPausedException("Download paused");
                            }
                        }
                    }
                }
            } else {
                Log.e(TAG, "downloadChunk error HfApiException " + response.code());
                throw new HfApiException("HTTP error: " + response.code());
            }
        } catch (IOException e) {
            Log.e(TAG, "downloadChunk error IOException", e);
            throw new HfApiException( "Connection error: " + e.getMessage());
        }
    }

    public interface FileDownloadListener {
        boolean onDownloadDelta(String fileName, long downloadedBytes, long totalBytes, long delta);
    }
}
