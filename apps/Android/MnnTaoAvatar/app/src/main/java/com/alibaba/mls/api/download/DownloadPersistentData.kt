// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mls.api.download

import android.content.Context
import com.alibaba.mls.api.HfFileMetadata
import com.alibaba.mls.api.download.DownloadFileUtils.getLastFileName
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken

object DownloadPersistentData {
    const val METADATA_KEY: String = "meta_data"

    const val SIZE_TOTAL_KEY: String = "size_total"

    const val SIZE_SAVED_KEY: String = "size_saved"


    fun saveMetaData(context: Context, modelId: String?, metaDataList: List<HfFileMetadata?>?) {
        var modelId = modelId
        modelId = getLastFileName(modelId)
        val sharedPreferences =
            context.getSharedPreferences("DOWNLOAD_$modelId", Context.MODE_PRIVATE)
        val editor = sharedPreferences.edit()
        val gson = Gson()
        val json = gson.toJson(metaDataList)
        editor.putString(METADATA_KEY, json)
        editor.apply()
    }

    @JvmStatic
    fun getMetaData(context: Context, modelId: String?): List<HfFileMetadata>? {
        var modelId = modelId
        modelId = getLastFileName(modelId)
        val sharedPreferences =
            context.getSharedPreferences("DOWNLOAD_$modelId", Context.MODE_PRIVATE)
        val json = sharedPreferences.getString(METADATA_KEY, null)
        if (json != null) {
            val gson = Gson()
            val type = object : TypeToken<List<HfFileMetadata?>?>() {}.type
            return gson.fromJson(json, type)
        }
        return null
    }

    @JvmStatic
    fun saveDownloadSizeTotal(context: Context, modelId: String?, total: Long) {
        var modelId = modelId
        modelId = getLastFileName(modelId)
        val sharedPreferences =
            context.getSharedPreferences("DOWNLOAD_$modelId", Context.MODE_PRIVATE)
        val editor = sharedPreferences.edit()
        editor.putLong(SIZE_TOTAL_KEY, total)
        editor.apply()
    }

    @JvmStatic
    fun getDownloadSizeTotal(context: Context, modelId: String?): Long {
        var modelId = modelId
        modelId = getLastFileName(modelId)
        val sharedPreferences =
            context.getSharedPreferences("DOWNLOAD_$modelId", Context.MODE_PRIVATE)
        return sharedPreferences.getLong(SIZE_TOTAL_KEY, 0)
    }

    @JvmStatic
    fun saveDownloadSizeSaved(context: Context, modelId: String?, saved: Long) {
        var modelId = modelId
        modelId = getLastFileName(modelId)
        val sharedPreferences =
            context.getSharedPreferences("DOWNLOAD_$modelId", Context.MODE_PRIVATE)
        val editor = sharedPreferences.edit()
        editor.putLong(SIZE_SAVED_KEY, saved)
        editor.apply()
    }

    @JvmStatic
    fun getDownloadSizeSaved(context: Context, modelId: String?): Long {
        var modelId = modelId
        modelId = getLastFileName(modelId)
        val sharedPreferences =
            context.getSharedPreferences("DOWNLOAD_$modelId", Context.MODE_PRIVATE)
        return sharedPreferences.getLong(SIZE_SAVED_KEY, 0)
    }

    @JvmStatic
    fun removeProgress(context: Context, modelId: String?) {
        var modelId = modelId
        modelId = getLastFileName(modelId)
        val sharedPreferences =
            context.getSharedPreferences("DOWNLOAD_$modelId", Context.MODE_PRIVATE)
        val editor = sharedPreferences.edit()
        editor.remove(SIZE_SAVED_KEY)
        editor.remove(SIZE_TOTAL_KEY)
        editor.apply()
    }
}
