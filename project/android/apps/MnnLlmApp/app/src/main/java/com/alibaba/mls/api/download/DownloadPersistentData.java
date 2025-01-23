// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mls.api.download;

import android.content.Context;
import android.content.SharedPreferences;

import com.alibaba.mls.api.HfFileMetadata;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

import java.lang.reflect.Type;
import java.util.List;

public class DownloadPersistentData {

    public static final String METADATA_KEY = "meta_data";

    public static final String SIZE_TOTAL_KEY = "size_total";

    public static final String SIZE_SAVED_KEY = "size_saved";


    public static void saveMetaData(Context context, String modelId, List<HfFileMetadata> metaDataList) {
        modelId = HfFileUtils.getLastFileName(modelId);
        SharedPreferences sharedPreferences = context.getSharedPreferences("DOWNLOAD_" + modelId, Context.MODE_PRIVATE);
        SharedPreferences.Editor editor = sharedPreferences.edit();
        Gson gson = new Gson();
        String json = gson.toJson(metaDataList);
        editor.putString(METADATA_KEY, json);
        editor.apply();
    }

    public static List<HfFileMetadata> getMetaData(Context context, String modelId) {
        modelId = HfFileUtils.getLastFileName(modelId);
        SharedPreferences sharedPreferences = context.getSharedPreferences("DOWNLOAD_" + modelId, Context.MODE_PRIVATE);
        String json = sharedPreferences.getString(METADATA_KEY, null);
        if (json != null) {
            Gson gson = new Gson();
            Type type = new TypeToken<List<HfFileMetadata>>() {}.getType();
            return gson.fromJson(json, type);
        }
        return null;
    }

    public static void saveDownloadSizeTotal(Context context, String modelId, long total) {
        modelId = HfFileUtils.getLastFileName(modelId);
        SharedPreferences sharedPreferences = context.getSharedPreferences("DOWNLOAD_" + modelId, Context.MODE_PRIVATE);
        SharedPreferences.Editor editor = sharedPreferences.edit();
        editor.putLong(SIZE_TOTAL_KEY, total);
        editor.apply();
    }

    public static long getDownloadSizeTotal(Context context, String modelId) {
        modelId = HfFileUtils.getLastFileName(modelId);
        SharedPreferences sharedPreferences = context.getSharedPreferences("DOWNLOAD_" + modelId, Context.MODE_PRIVATE);
        return sharedPreferences.getLong(SIZE_TOTAL_KEY, 0);
    }

    public static void saveDownloadSizeSaved(Context context, String modelId, long saved) {
        modelId = HfFileUtils.getLastFileName(modelId);
        SharedPreferences sharedPreferences = context.getSharedPreferences("DOWNLOAD_" + modelId, Context.MODE_PRIVATE);
        SharedPreferences.Editor editor = sharedPreferences.edit();
        editor.putLong(SIZE_SAVED_KEY, saved);
        editor.apply();
    }

    public static long getDownloadSizeSaved(Context context, String modelId) {
        modelId = HfFileUtils.getLastFileName(modelId);
        SharedPreferences sharedPreferences = context.getSharedPreferences("DOWNLOAD_" + modelId, Context.MODE_PRIVATE);
        return sharedPreferences.getLong(SIZE_SAVED_KEY, 0);
    }

    public static void removeProgress(Context context, String modelId) {
        modelId = HfFileUtils.getLastFileName(modelId);
        SharedPreferences sharedPreferences = context.getSharedPreferences("DOWNLOAD_" + modelId, Context.MODE_PRIVATE);
        SharedPreferences.Editor editor = sharedPreferences.edit();
        editor.remove(SIZE_SAVED_KEY);
        editor.remove(SIZE_TOTAL_KEY);
        editor.apply();
    }
}
