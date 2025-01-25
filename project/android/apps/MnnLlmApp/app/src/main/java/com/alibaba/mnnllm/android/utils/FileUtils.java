// Created by ruoyi.sjd on 2025/1/10.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.utils;

import android.content.Context;
import android.media.MediaMetadataRetriever;
import android.net.Uri;
import android.util.Log;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;

public class FileUtils {

    public static final String TAG = "FileUtils";
    public static long getAudioDuration(String audioFilePath) {
        MediaMetadataRetriever mmr = new MediaMetadataRetriever();
        try {
            mmr.setDataSource(audioFilePath);
            String durationStr = mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION);
            return durationStr != null ? Long.parseLong(durationStr) / 1000 : -1;
        } catch (Exception e) {
            Log.e(TAG, "", e);
            return 1;
        } finally {
            try {
                mmr.release();
            } catch (IOException e) {
                Log.e(TAG, "", e);
            }
        }
    }

    public static String generateDestDiffusionFilePath(Context context, String sessionId) {
        return generateDestFilePathKindOf(context, sessionId, "diffusion", "jpg");
    }

    public static String generateDestPhotoFilePath(Context context, String sessionId) {
        return generateDestFilePathKindOf(context, sessionId, "photo", "jpg");
    }

    public static String generateDestAudioFilePath(Context context, String sessionId) {
        return generateDestFilePathKindOf(context, sessionId, "audio", "wav");
    }

    public static String generateDestRecordFilePath(Context context, String sessionId) {
        return generateDestFilePathKindOf(context, sessionId, "record", "wav");
    }

    public static String generateDestImageFilePath(Context context, String sessionId) {
        return generateDestFilePathKindOf(context, sessionId, "image", "jpg");
    }

    private static String generateDestFilePathKindOf(Context context, String sessionId, String kind, String extension) {
        String path = context.getFilesDir().getAbsolutePath() + "/" + sessionId + "/" + kind + "_" + System.currentTimeMillis() + "." + extension;
        ensureParentDirectoriesExist(new File(path));
        return path;
    }

    public static String getSessionResourceBasePath(Context context, String sessionId) {
        return context.getFilesDir().getAbsolutePath() + "/" + sessionId;
    }

    public static void ensureParentDirectoriesExist(File file) {
        File parentDir = file.getParentFile();
        if (parentDir != null && !parentDir.exists()) {
            parentDir.mkdirs();
        }
    }

    public static File copyFileUriToPath(Context context, Uri fileUri, String destFilePath) throws IOException {
        InputStream inputStream = null;
        OutputStream outputStream = null;
        try {
            // Open an InputStream from the Uri
            inputStream = context.getContentResolver().openInputStream(fileUri);
            if (inputStream == null) {
                throw new IllegalArgumentException("Unable to open InputStream from Uri");
            }
            // Create the destination file
            File destinationFile = new File(destFilePath);
            ensureParentDirectoriesExist(destinationFile);
            outputStream = Files.newOutputStream(destinationFile.toPath());

            // Buffer for data transfer
            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = inputStream.read(buffer)) != -1) {
                outputStream.write(buffer, 0, bytesRead);
            }
            outputStream.flush();

            return destinationFile;
        }  finally {
            try {
                if (inputStream != null) inputStream.close();
            } catch (Exception ignored) {}
            try {
                if (outputStream != null) outputStream.close();
            } catch (Exception ignored) {}
        }
    }


}

