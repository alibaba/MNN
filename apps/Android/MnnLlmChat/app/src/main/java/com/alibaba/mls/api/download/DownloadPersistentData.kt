// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mls.api.download

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.longPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import com.alibaba.mls.api.download.DownloadFileUtils.getLastFileName
import com.alibaba.mnnllm.android.model.ModelUtils
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.runBlocking
import androidx.datastore.preferences.core.stringPreferencesKey
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import com.alibaba.mls.api.hf.HfFileMetadata
import java.io.File

// Single DataStore instance for all download data
private val Context.downloadDataStore: DataStore<Preferences> by preferencesDataStore(name = "download_data")

object DownloadPersistentData {
    const val METADATA_KEY: String = "meta_data"
    const val SIZE_TOTAL_KEY: String = "size_total"
    const val SIZE_SAVED_KEY: String = "size_saved"
    const val SIZE_MARKET_TOTAL_KEY: String = "size_market_total"
    const val DOWNLOADED_TIME_KEY: String = "downloaded_time"

    // Create preference keys with modelId
    private fun createSizeTotalKey(modelId: String): Preferences.Key<Long> = 
        longPreferencesKey("${SIZE_TOTAL_KEY}_$modelId")
    
    private fun createSizeSavedKey(modelId: String): Preferences.Key<Long> = 
        longPreferencesKey("${SIZE_SAVED_KEY}_$modelId")
    
    private fun createMetaDataKey(modelId: String): Preferences.Key<String> = 
        stringPreferencesKey("${METADATA_KEY}_$modelId")

    private fun createSizeMarketTotalKey(modelId: String): Preferences.Key<Long> = 
        longPreferencesKey("${SIZE_MARKET_TOTAL_KEY}_$modelId")

    private fun createDownloadedTimeKey(modelId: String): Preferences.Key<Long> = 
        longPreferencesKey("${DOWNLOADED_TIME_KEY}_$modelId")

    fun saveDownloadSizeTotal(context: Context, modelId: String, total: Long) {
        runBlocking { saveDownloadSizeTotalSuspend(context, modelId, total) }
    }
    
    suspend fun saveDownloadSizeTotalSuspend(context: Context, modelId: String, total: Long) {
        val normalizedModelId = getLastFileName(modelId)
        val key = createSizeTotalKey(normalizedModelId)
        
        context.downloadDataStore.edit { preferences ->
            preferences[key] = total
        }
    }

    
    // Synchronous versions for backward compatibility
    @JvmStatic
    fun getDownloadSizeTotal(context: Context, modelId: String): Long {
        return runBlocking { getDownloadSizeTotalSuspend(context, modelId) }
    }
    
    suspend fun getDownloadSizeTotalSuspend(context: Context, modelId: String): Long {
        val normalizedModelId = getLastFileName(modelId)
        val key = createSizeTotalKey(normalizedModelId)
        
        // First try to read from DataStore
        val dataStoreValue = context.downloadDataStore.data
            .map { preferences -> preferences[key] }
            .first()
        
        if (dataStoreValue != null) {
            return dataStoreValue
        }
        
        // If DataStore doesn't have data, try to migrate from SharedPreferences
        return migrateFromSharedPrefsTotal(context, normalizedModelId, key)
    }

    fun saveDownloadSizeSaved(context: Context, modelId: String, saved: Long) {
        runBlocking { saveDownloadSizeSavedSuspend(context, modelId, saved) }
    }
    
    suspend fun saveDownloadSizeSavedSuspend(context: Context, modelId: String, saved: Long) {
        val newModelId = ModelUtils.safeModelId(modelId)
        val key = createSizeSavedKey(newModelId)
        
        context.downloadDataStore.edit { preferences ->
            preferences[key] = saved
        }
    }

    fun getDownloadSizeSaved(context: Context, modelId: String): Long {
        return runBlocking { getDownloadSizeSavedSuspend(context, modelId) }
    }
    
    suspend fun getDownloadSizeSavedSuspend(context: Context, modelId: String): Long {
        val newModelId = ModelUtils.safeModelId(modelId)
        val key = createSizeSavedKey(newModelId)
        
        // First try to read from DataStore
        val dataStoreValue = context.downloadDataStore.data
            .map { preferences -> preferences[key] }
            .first()
        
        if (dataStoreValue != null) {
            return dataStoreValue
        }
        
        // If DataStore doesn't have data, try to migrate from SharedPreferences
        return migrateFromSharedPrefsSaved(context, newModelId, key)
    }

    fun removeProgress(context: Context, modelId: String) {
        runBlocking { removeProgressSuspend(context, modelId) }
    }
    
    suspend fun removeProgressSuspend(context: Context, modelId: String) {
        val normalizedModelId = ModelUtils.safeModelId(modelId)
        val sizeSavedKey = createSizeSavedKey(normalizedModelId)
        val downloadedTimeKey = createDownloadedTimeKey(normalizedModelId)
        
        // Remove from DataStore
        context.downloadDataStore.edit { preferences ->
            preferences.remove(sizeSavedKey)
            preferences.remove(downloadedTimeKey)
        }
        
        // Also remove from SharedPreferences (cleanup) - removeProgress originally used getLastFileName
        val oldModelIdKey = getLastFileName(modelId)
        val sharedPreferences = context.getSharedPreferences("DOWNLOAD_$oldModelIdKey", Context.MODE_PRIVATE)
        sharedPreferences.edit().remove(SIZE_SAVED_KEY).apply()
        
        // Check if SharedPreferences file is empty and delete if so
        deleteSharedPrefsFileIfEmpty(context, "DOWNLOAD_$oldModelIdKey")
    }
    
    fun saveMetaData(context: Context, modelId: String, metaData: Map<String, HfFileMetadata>) {
        runBlocking { saveMetaDataSuspend(context, modelId, metaData) }
    }
    
    suspend fun saveMetaDataSuspend(context: Context, modelId: String, metaData: Map<String, HfFileMetadata>) {
        val normalizedModelId = ModelUtils.safeModelId(modelId)
        val key = createMetaDataKey(normalizedModelId)
        val json = Gson().toJson(metaData)
        
        context.downloadDataStore.edit { preferences ->
            preferences[key] = json
        }
    }
    
    fun getMetaData(context: Context, modelId: String): Map<String, HfFileMetadata>? {
        return runBlocking { getMetaDataSuspend(context, modelId) }
    }
    
    suspend fun getMetaDataSuspend(context: Context, modelId: String): Map<String, HfFileMetadata>? {
        val normalizedModelId = ModelUtils.safeModelId(modelId)
        val key = createMetaDataKey(normalizedModelId)
        
        // First try to read from DataStore
        val dataStoreValue = context.downloadDataStore.data
            .map { preferences -> preferences[key] }
            .first()
        
        if (dataStoreValue != null) {
            return try {
                val type = object : TypeToken<Map<String, HfFileMetadata>>() {}.type
                Gson().fromJson(dataStoreValue, type)
            } catch (e: Exception) {
                null
            }
        }
        
        // If DataStore doesn't have data, try to migrate from SharedPreferences
        return migrateMetaDataFromSharedPrefs(context, normalizedModelId, key)
    }
    
    fun saveDownloadedTime(context: Context, modelId: String, downloadedTime: Long) {
        runBlocking { saveDownloadedTimeSuspend(context, modelId, downloadedTime) }
    }
    
    suspend fun saveDownloadedTimeSuspend(context: Context, modelId: String, downloadedTime: Long) {
        val normalizedModelId = ModelUtils.safeModelId(modelId)
        val key = createDownloadedTimeKey(normalizedModelId)
        
        context.downloadDataStore.edit { preferences ->
            preferences[key] = downloadedTime
        }
    }

    fun getDownloadedTime(context: Context, modelId: String): Long {
        return runBlocking { getDownloadedTimeSuspend(context, modelId) }
    }
    
    suspend fun getDownloadedTimeSuspend(context: Context, modelId: String): Long {
        val normalizedModelId = ModelUtils.safeModelId(modelId)
        val key = createDownloadedTimeKey(normalizedModelId)
        
        // Try to read from DataStore
        val dataStoreValue = context.downloadDataStore.data
            .map { preferences -> preferences[key] }
            .first()
        
        return dataStoreValue ?: 0L
    }

    fun saveMarketSizeTotal(context: Context, modelId: String, total: Long) {
        runBlocking { saveMarketSizeTotalSuspend(context, modelId, total) }
    }
    
    suspend fun saveMarketSizeTotalSuspend(context: Context, modelId: String, total: Long) {
        val normalizedModelId = ModelUtils.safeModelId(modelId)
        val key = createSizeMarketTotalKey(normalizedModelId)
        
        context.downloadDataStore.edit { preferences ->
            preferences[key] = total
        }
    }

    fun getMarketSizeTotal(context: Context, modelId: String): Long {
        return runBlocking { getMarketSizeTotalSuspend(context, modelId) }
    }
    
    suspend fun getMarketSizeTotalSuspend(context: Context, modelId: String): Long {
        val normalizedModelId = ModelUtils.safeModelId(modelId)
        val key = createSizeMarketTotalKey(normalizedModelId)
        
        // Try to read from DataStore
        val dataStoreValue = context.downloadDataStore.data
            .map { preferences -> preferences[key] }
            .first()
        
        return dataStoreValue ?: 0L
    }
    
    // Private migration helpers
    private fun deleteSharedPrefsFileIfEmpty(context: Context, preferencesName: String) {
        val sharedPrefs = context.getSharedPreferences(preferencesName, Context.MODE_PRIVATE)
        val allKeys = sharedPrefs.all
        
        // If SharedPreferences contains no keys, delete the file
        if (allKeys.isEmpty()) {
            try {
                // Get the SharedPreferences file path and delete it
                val prefsDir = File(context.applicationInfo.dataDir, "shared_prefs")
                val prefsFile = File(prefsDir, "$preferencesName.xml")
                if (prefsFile.exists()) {
                    prefsFile.delete()
                }
            } catch (e: Exception) {
                // Ignore errors during file deletion
            }
        }
    }
    private suspend fun migrateFromSharedPrefsTotal(
        context: Context, 
        modelId: String, 
        datastoreKey: Preferences.Key<Long>
    ): Long {
        // For total size, the original implementation used getLastFileName
        val oldModelIdKey = getLastFileName(modelId)
        val sharedPreferences = context.getSharedPreferences("DOWNLOAD_$oldModelIdKey", Context.MODE_PRIVATE)
        val sharedPrefValue = sharedPreferences.getLong(SIZE_TOTAL_KEY, 0)
        
        if (sharedPrefValue != 0L) {
            // Migrate to DataStore
            context.downloadDataStore.edit { preferences ->
                preferences[datastoreKey] = sharedPrefValue
            }
            
            // Remove from SharedPreferences after successful migration
            sharedPreferences.edit().remove(SIZE_TOTAL_KEY).apply()
            
            // Check if SharedPreferences file is empty and delete if so
            deleteSharedPrefsFileIfEmpty(context, "DOWNLOAD_$oldModelIdKey")
            
            return sharedPrefValue
        }
        
        return 0L
    }
    
    private suspend fun migrateFromSharedPrefsSaved(
        context: Context, 
        modelId: String, 
        datastoreKey: Preferences.Key<Long>
    ): Long {
        // For saved size, the original implementation used ModelUtils.safeModelId - same as current
        val sharedPreferences = context.getSharedPreferences("DOWNLOAD_$modelId", Context.MODE_PRIVATE)
        val sharedPrefValue = sharedPreferences.getLong(SIZE_SAVED_KEY, -1)
        
        if (sharedPrefValue != -1L) {
            // Migrate to DataStore
            context.downloadDataStore.edit { preferences ->
                preferences[datastoreKey] = sharedPrefValue
            }
            
            // Remove from SharedPreferences after successful migration
            sharedPreferences.edit().remove(SIZE_SAVED_KEY).apply()
            
            // Check if SharedPreferences file is empty and delete if so
            deleteSharedPrefsFileIfEmpty(context, "DOWNLOAD_$modelId")
            
            return sharedPrefValue
        }
        
        return -1L
    }
    
    private suspend fun migrateMetaDataFromSharedPrefs(
        context: Context, 
        modelId: String, 
        datastoreKey: Preferences.Key<String>
    ): Map<String, HfFileMetadata>? {
        // For metadata, assume it used the same pattern as removeProgress (getLastFileName)
        val oldModelIdKey = getLastFileName(modelId)
        val sharedPreferences = context.getSharedPreferences("DOWNLOAD_$oldModelIdKey", Context.MODE_PRIVATE)
        val sharedPrefValue = sharedPreferences.getString(METADATA_KEY, null)
        
        if (sharedPrefValue != null) {
            try {
                // Migrate to DataStore
                context.downloadDataStore.edit { preferences ->
                    preferences[datastoreKey] = sharedPrefValue
                }
                
                // Remove from SharedPreferences after successful migration
                sharedPreferences.edit().remove(METADATA_KEY).apply()
                
                // Check if SharedPreferences file is empty and delete if so
                deleteSharedPrefsFileIfEmpty(context, "DOWNLOAD_$oldModelIdKey")
                
                // Parse and return the data
                val type = object : TypeToken<Map<String, HfFileMetadata>>() {}.type
                return Gson().fromJson(sharedPrefValue, type)
            } catch (e: Exception) {
                return null
            }
        }
        
        return null
    }
}
