// Created by ruoyi.sjd on 2025/01/05.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.chat.model

import android.annotation.SuppressLint
import android.content.ContentValues
import android.content.Context
import android.net.Uri
import android.text.TextUtils
import android.util.Log
import org.json.JSONArray

class ChatDataManager private constructor(context: Context) {
    private val dbHelper = ChatDatabaseHelper(context)
    
    init {
        ensureLastChatTimeColumn()
        ensureSessionModeColumn()
        ensureAgentTables()
        fixAllMissingLastChatTimes()
    }

    fun addOrUpdateSession(
        sessionId: String,
        modelId: String?,
        sessionMode: String? = null
    ) {
        Log.d(TAG, "addOrUpdateSession: sessionId: $sessionId modelId: $modelId sessionMode: $sessionMode")
        val db = dbHelper.writableDatabase

        val cursor = db.query(
            ChatDatabaseHelper.TABLE_SESSION,
            arrayOf(ChatDatabaseHelper.COLUMN_SESSION_ID),
            ChatDatabaseHelper.COLUMN_SESSION_ID + "=?",
            arrayOf(sessionId), null, null, null
        )

        val exists = cursor.moveToFirst()
        cursor.close()

        val values = ContentValues()
        values.put(ChatDatabaseHelper.COLUMN_SESSION_ID, sessionId)
        values.put(ChatDatabaseHelper.COLUMN_MODEL_ID, modelId)
        if (!sessionMode.isNullOrBlank()) {
            values.put(ChatDatabaseHelper.COLUMN_SESSION_MODE, normalizeSessionMode(sessionMode))
        }
        
        if (exists) {
            // Don't update lastChatTime when just updating session info
            db.update(
                ChatDatabaseHelper.TABLE_SESSION, values,
                ChatDatabaseHelper.COLUMN_SESSION_ID + "=?",
                arrayOf(sessionId)
            )
        } else {
            // Set initial lastChatTime to 0 for new sessions
            values.put(ChatDatabaseHelper.COLUMN_LAST_CHAT_TIME, 0L)
            if (sessionMode.isNullOrBlank()) {
                values.put(ChatDatabaseHelper.COLUMN_SESSION_MODE, ChatDatabaseHelper.SESSION_MODE_NORMAL)
            }
            db.insert(ChatDatabaseHelper.TABLE_SESSION, null, values)
        }
        db.close()
    }

    fun updateSessionMode(sessionId: String, sessionMode: String) {
        val db = dbHelper.writableDatabase
        val values = ContentValues()
        values.put(ChatDatabaseHelper.COLUMN_SESSION_MODE, normalizeSessionMode(sessionMode))
        db.update(
            ChatDatabaseHelper.TABLE_SESSION,
            values,
            ChatDatabaseHelper.COLUMN_SESSION_ID + "=?",
            arrayOf(sessionId)
        )
        db.close()
    }

    @SuppressLint("Range")
    fun getSessionMode(sessionId: String): String {
        val db = dbHelper.readableDatabase
        return try {
            val cursor = db.query(
                ChatDatabaseHelper.TABLE_SESSION,
                arrayOf(ChatDatabaseHelper.COLUMN_SESSION_MODE),
                ChatDatabaseHelper.COLUMN_SESSION_ID + "=?",
                arrayOf(sessionId), null, null, null
            )
            val mode = if (cursor.moveToFirst()) {
                cursor.getString(cursor.getColumnIndex(ChatDatabaseHelper.COLUMN_SESSION_MODE))
            } else {
                null
            }
            cursor.close()
            normalizeSessionMode(mode)
        } catch (e: Exception) {
            ChatDatabaseHelper.SESSION_MODE_NORMAL
        } finally {
            db.close()
        }
    }

    @SuppressLint("Range")
    fun getAgentMemories(limit: Int = 40): List<AgentMemoryItem> {
        val list = mutableListOf<AgentMemoryItem>()
        val db = dbHelper.readableDatabase
        try {
            val cursor = db.query(
                ChatDatabaseHelper.TABLE_AGENT_MEMORY,
                null,
                null,
                null,
                null,
                null,
                ChatDatabaseHelper.COLUMN_AGENT_MEMORY_UPDATED_AT + " DESC",
                limit.coerceAtLeast(1).toString()
            )
            while (cursor.moveToNext()) {
                list += AgentMemoryItem(
                    id = cursor.getLong(cursor.getColumnIndex(ChatDatabaseHelper.COLUMN_AGENT_MEMORY_ID)),
                    category = cursor.getString(cursor.getColumnIndex(ChatDatabaseHelper.COLUMN_AGENT_MEMORY_CATEGORY)).orEmpty(),
                    content = cursor.getString(cursor.getColumnIndex(ChatDatabaseHelper.COLUMN_AGENT_MEMORY_CONTENT)).orEmpty(),
                    source = cursor.getString(cursor.getColumnIndex(ChatDatabaseHelper.COLUMN_AGENT_MEMORY_SOURCE)).orEmpty(),
                    updatedAt = cursor.getLong(cursor.getColumnIndex(ChatDatabaseHelper.COLUMN_AGENT_MEMORY_UPDATED_AT))
                )
            }
            cursor.close()
        } catch (e: Exception) {
            Log.e(TAG, "getAgentMemories failed", e)
        } finally {
            db.close()
        }
        return list
    }

    @SuppressLint("Range")
    fun getEnabledAgentSkills(): List<AgentSkillItem> {
        val list = mutableListOf<AgentSkillItem>()
        val db = dbHelper.readableDatabase
        try {
            val cursor = db.query(
                ChatDatabaseHelper.TABLE_AGENT_SKILL,
                null,
                ChatDatabaseHelper.COLUMN_AGENT_SKILL_ENABLED + "=1",
                null,
                null,
                null,
                ChatDatabaseHelper.COLUMN_AGENT_SKILL_CREATED_AT + " DESC"
            )
            while (cursor.moveToNext()) {
                list += AgentSkillItem(
                    id = cursor.getLong(cursor.getColumnIndex(ChatDatabaseHelper.COLUMN_AGENT_SKILL_ID)),
                    name = cursor.getString(cursor.getColumnIndex(ChatDatabaseHelper.COLUMN_AGENT_SKILL_NAME)).orEmpty(),
                    description = cursor.getString(cursor.getColumnIndex(ChatDatabaseHelper.COLUMN_AGENT_SKILL_DESCRIPTION)).orEmpty(),
                    triggerKeywords = cursor.getString(cursor.getColumnIndex(ChatDatabaseHelper.COLUMN_AGENT_SKILL_TRIGGER_KEYWORDS)).orEmpty(),
                    actionTemplate = cursor.getString(cursor.getColumnIndex(ChatDatabaseHelper.COLUMN_AGENT_SKILL_ACTION_TEMPLATE)).orEmpty(),
                    enabled = cursor.getInt(cursor.getColumnIndex(ChatDatabaseHelper.COLUMN_AGENT_SKILL_ENABLED)) == 1,
                    createdAt = cursor.getLong(cursor.getColumnIndex(ChatDatabaseHelper.COLUMN_AGENT_SKILL_CREATED_AT))
                )
            }
            cursor.close()
        } catch (e: Exception) {
            Log.e(TAG, "getEnabledAgentSkills failed", e)
        } finally {
            db.close()
        }
        return list
    }

    fun upsertAgentMemory(category: String?, content: String?, source: String = "agent") {
        val safeCategory = compactAgentText(category, 48)
        val safeContent = compactAgentText(content, MEMORY_CONTENT_MAX_CHARS)
        if (safeCategory.isBlank() || safeContent.isBlank()) return
        if (!isUsefulAgentMemory(safeCategory, safeContent)) return
        val db = dbHelper.writableDatabase
        try {
            val values = ContentValues().apply {
                put(ChatDatabaseHelper.COLUMN_AGENT_MEMORY_CATEGORY, safeCategory)
                put(ChatDatabaseHelper.COLUMN_AGENT_MEMORY_CONTENT, safeContent)
                put(ChatDatabaseHelper.COLUMN_AGENT_MEMORY_SOURCE, source)
                put(ChatDatabaseHelper.COLUMN_AGENT_MEMORY_UPDATED_AT, System.currentTimeMillis())
            }
            db.insert(ChatDatabaseHelper.TABLE_AGENT_MEMORY, null, values)
        } catch (e: Exception) {
            Log.e(TAG, "upsertAgentMemory failed", e)
        } finally {
            db.close()
        }
    }

    fun upsertAgentSkill(
        name: String?,
        description: String?,
        triggerKeywords: List<String>?,
        actionTemplate: String?
    ) {
        val safeName = compactAgentText(name, 64)
        val safeTemplate = compactAgentText(actionTemplate, SKILL_ACTION_MAX_CHARS)
        if (safeName.isBlank() || safeTemplate.isBlank()) return
        if (!isUsefulAgentSkill(safeName, safeTemplate)) return
        val db = dbHelper.writableDatabase
        try {
            val values = ContentValues().apply {
                put(ChatDatabaseHelper.COLUMN_AGENT_SKILL_NAME, safeName)
                put(ChatDatabaseHelper.COLUMN_AGENT_SKILL_DESCRIPTION, compactAgentText(description, MEMORY_CONTENT_MAX_CHARS))
                put(ChatDatabaseHelper.COLUMN_AGENT_SKILL_TRIGGER_KEYWORDS, JSONArray(triggerKeywords.orEmpty().map { compactAgentText(it, 32) }.filter { it.isNotBlank() }).toString())
                put(ChatDatabaseHelper.COLUMN_AGENT_SKILL_ACTION_TEMPLATE, safeTemplate)
                put(ChatDatabaseHelper.COLUMN_AGENT_SKILL_ENABLED, 1)
                put(ChatDatabaseHelper.COLUMN_AGENT_SKILL_CREATED_AT, System.currentTimeMillis())
            }
            db.replace(ChatDatabaseHelper.TABLE_AGENT_SKILL, null, values)
        } catch (e: Exception) {
            Log.e(TAG, "upsertAgentSkill failed", e)
        } finally {
            db.close()
        }
    }

    fun buildAgentMemoryBlock(limit: Int = 20): String {
        return getAgentMemories(limit)
            .filter { it.content.isNotBlank() }
            .filter { isUsefulAgentMemory(it.category, it.content) }
            .joinToString("\n") {
                "- ${compactAgentText(it.category, 48)}: ${compactAgentText(it.content, MEMORY_CONTENT_MAX_CHARS)}"
            }
    }

    fun buildAgentSkillBlock(): String {
        return getEnabledAgentSkills()
            .filter { isUsefulAgentSkill(it.name, it.actionTemplate) }
            .joinToString("\n\n") { skill ->
                buildString {
                    appendLine("- ${compactAgentText(skill.name, 64)}: ${compactAgentText(skill.description, MEMORY_CONTENT_MAX_CHARS)}")
                    appendLine("  triggers: ${compactAgentText(skill.triggerKeywords, MEMORY_CONTENT_MAX_CHARS)}")
                    append("  action: ${compactAgentText(skill.actionTemplate, SKILL_ACTION_MAX_CHARS)}")
                }
            }
    }

    fun runLocalAgentSkills(userInput: String): String {
        val normalizedInput = userInput.lowercase()
        val outputs = mutableListOf<String>()
        for (skill in getEnabledAgentSkills()) {
            val triggers = parseTriggerKeywords(skill.triggerKeywords)
            val matched = triggers.any { keyword ->
                keyword.isNotBlank() && normalizedInput.contains(keyword.lowercase())
            }
            if (matched) {
                outputs += "【${skill.name}】${skill.actionTemplate}"
            }
        }
        return outputs.joinToString("\n")
    }

    private fun compactAgentText(value: String?, maxChars: Int): String {
        val compacted = value.orEmpty()
            .replace(Regex("\\s+"), " ")
            .trim()
        if (compacted.length <= maxChars) return compacted
        return compacted.take(maxChars).trimEnd() + "..."
    }

    private fun isUsefulAgentMemory(category: String?, content: String?): Boolean {
        val text = "${category.orEmpty()} ${content.orEmpty()}".trim()
        if (text.length < 8) return false
        if (looksLikeTaskNoise(text)) return false
        val lower = text.lowercase()
        return listOf(
            "用户", "偏好", "喜欢", "不喜欢", "希望", "默认", "记住", "姓名", "学校", "课程", "专业",
            "preference", "prefer", "user", "remember", "default", "likes", "dislikes"
        ).any { lower.contains(it.lowercase()) }
    }

    private fun isUsefulAgentSkill(name: String?, actionTemplate: String?): Boolean {
        val text = "${name.orEmpty()} ${actionTemplate.orEmpty()}".trim()
        if (text.length < 12) return false
        if (looksLikeTaskNoise(text)) return false
        return actionTemplate.orEmpty().contains(
            Regex("搜索|浏览|读取|分析|生成|保存|检查|search|browse|read|analy[sz]e|generate|save|check", RegexOption.IGNORE_CASE)
        )
    }

    private fun looksLikeTaskNoise(text: String): Boolean {
        val lower = text.lowercase()
        return listOf(
            "已准备", "已经准备", "已经修正", "成功创建", "用于展示", "希望这个例子", "随时告诉我",
            "现在我已经", "我将为您", "请稍等", "示例数据", "sample_data", "sample.xlsx",
            "prepared", "successfully created", "example data", "ready to", "feel free"
        ).any { lower.contains(it.lowercase()) }
    }

    private fun parseTriggerKeywords(raw: String): List<String> {
        return runCatching {
            val array = JSONArray(raw)
            (0 until array.length()).map { array.optString(it) }
        }.getOrElse {
            raw.split(',', '，', ';', '；').map { it.trim() }
        }.filter { it.isNotBlank() }
    }

    fun addChatData(sessionId: String?, chatDataItem: ChatDataItem) {
        if (sessionId.isNullOrEmpty()) {
            Log.e(TAG, "addChatData: sessionId is null or empty")
            return
        }
        
        if (chatDataItem.text.isNullOrEmpty() && chatDataItem.imageUris.isNullOrEmpty() && chatDataItem.audioUri == null && chatDataItem.videoUri == null) {
            Log.w(TAG, "addChatData: chatDataItem has no content to save")
            return
        }
        
        Log.d(TAG, "addChatData: sessionId=$sessionId, type=${chatDataItem.type}, textLength=${chatDataItem.text?.length ?: 0}, hasImage=${!chatDataItem.imageUris.isNullOrEmpty()}, hasVideo=${chatDataItem.videoUri != null}")
        
        val db = dbHelper.writableDatabase
        try {
            db.beginTransaction()
            
            // Insert chat data
            val values = ContentValues()
            values.put(ChatDatabaseHelper.COLUMN_SESSION_ID, sessionId)
            values.put(ChatDatabaseHelper.COLUMN_TIME, chatDataItem.time)
            values.put(ChatDatabaseHelper.COLUMN_TEXT, chatDataItem.text)
            values.put(ChatDatabaseHelper.COLUMN_TYPE, chatDataItem.type)
            if (!chatDataItem.imageUris.isNullOrEmpty()) {
                val jsonArray = JSONArray()
                chatDataItem.imageUris!!.forEach { uri ->
                    jsonArray.put(uri.toString())
                }
                values.put(ChatDatabaseHelper.COLUMN_IMAGE_URI, jsonArray.toString())
            } else {
                values.put(ChatDatabaseHelper.COLUMN_IMAGE_URI, null as String?)
            }
            if (chatDataItem.audioUri != null) {
                values.put(ChatDatabaseHelper.COLUMN_AUDIO_URI, chatDataItem.audioUri.toString())
            } else {
                values.put(ChatDatabaseHelper.COLUMN_AUDIO_URI, null as String?)
            }
            if (chatDataItem.videoUri != null) {
                values.put(ChatDatabaseHelper.COLUMN_VIDEO_URI, chatDataItem.videoUri.toString())
            } else {
                values.put(ChatDatabaseHelper.COLUMN_VIDEO_URI, null as String?)
            }
            values.put(ChatDatabaseHelper.COLUMN_AUDIO_DURATION, chatDataItem.audioDuration)
            values.put(ChatDatabaseHelper.COLUMN_DISPLAY_TEXT, chatDataItem.displayText)
            values.put(ChatDatabaseHelper.COLUMN_THINKING_TEXT, chatDataItem.thinkingText)
            values.put(ChatDatabaseHelper.COLUMN_THINKING_FINISHED_TIME, chatDataItem.thinkingFinishedTime)
            
            val rowId = db.insert(ChatDatabaseHelper.TABLE_CHAT, null, values)
            if (rowId == -1L) {
                Log.e(TAG, "addChatData: Failed to insert chat data")
                return
            }
            
            // Update session's lastChatTime
            val currentTime = System.currentTimeMillis()
            val sessionValues = ContentValues()
            sessionValues.put(ChatDatabaseHelper.COLUMN_LAST_CHAT_TIME, currentTime)
            val updatedRows = db.update(
                ChatDatabaseHelper.TABLE_SESSION,
                sessionValues,
                ChatDatabaseHelper.COLUMN_SESSION_ID + "=?",
                arrayOf(sessionId)
            )
            
            if (updatedRows == 0) {
                Log.w(TAG, "addChatData: No session found to update lastChatTime for sessionId=$sessionId")
            }
            
            db.setTransactionSuccessful()
            Log.d(TAG, "addChatData: Successfully saved chat data with rowId=$rowId")
            
        } catch (e: Exception) {
            Log.e(TAG, "addChatData: Database error for sessionId=$sessionId", e)
        } finally {
            try {
                db.endTransaction()
            } catch (e: Exception) {
                Log.e(TAG, "addChatData: Error ending transaction", e)
            }
            try {
                db.close()
            } catch (e: Exception) {
                Log.e(TAG, "addChatData: Error closing database", e)
            }
        }
    }

    @SuppressLint("Range")
    fun getChatDataBySession(sessionId: String): List<ChatDataItem> {
        val chatDataItemList: MutableList<ChatDataItem> = ArrayList()
        val db = dbHelper.readableDatabase

        val cursor = db.query(
            ChatDatabaseHelper.TABLE_CHAT,
            null,
            ChatDatabaseHelper.COLUMN_SESSION_ID + "=?",
            arrayOf(sessionId), null, null, null
        )
        while (cursor.moveToNext()) {
            val time = cursor.getString(cursor.getColumnIndex(ChatDatabaseHelper.COLUMN_TIME))
            val type = cursor.getInt(cursor.getColumnIndex(ChatDatabaseHelper.COLUMN_TYPE))
            val text = cursor.getString(cursor.getColumnIndex(ChatDatabaseHelper.COLUMN_TEXT))
            val imageUriStr =
                cursor.getString(cursor.getColumnIndex(ChatDatabaseHelper.COLUMN_IMAGE_URI))
            val audioUriStr =
                cursor.getString(cursor.getColumnIndex(ChatDatabaseHelper.COLUMN_AUDIO_URI))
            val videoUriStr =
                cursor.getString(cursor.getColumnIndex(ChatDatabaseHelper.COLUMN_VIDEO_URI))
            val audioDuration =
                cursor.getFloat(cursor.getColumnIndex(ChatDatabaseHelper.COLUMN_AUDIO_DURATION))
            val displayText =
                cursor.getString(cursor.getColumnIndex(ChatDatabaseHelper.COLUMN_DISPLAY_TEXT))
            val chatDataItem = ChatDataItem(time, type, text)
            
            if (!TextUtils.isEmpty(imageUriStr)) {
                try {
                    if (imageUriStr.startsWith("[")) {
                        val jsonArray = JSONArray(imageUriStr)
                        val uris = mutableListOf<Uri>()
                        for (i in 0 until jsonArray.length()) {
                            uris.add(Uri.parse(jsonArray.getString(i)))
                        }
                        chatDataItem.imageUris = uris
                    } else {
                        chatDataItem.imageUris = listOf(Uri.parse(imageUriStr))
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Error parsing image URI: $imageUriStr", e)
                    // Fallback to single URI if parsing failed but it's not null
                    if (!TextUtils.isEmpty(imageUriStr)) {
                         chatDataItem.imageUris = listOf(Uri.parse(imageUriStr))
                    }
                }
            }
            
            if (!TextUtils.isEmpty(displayText)) {
                chatDataItem.displayText = displayText
            }
            if (audioUriStr != null) {
                chatDataItem.audioUri = Uri.parse(audioUriStr)
                chatDataItem.audioDuration = audioDuration
            }
            if (videoUriStr != null) {
                chatDataItem.videoUri = Uri.parse(videoUriStr)
            }
            val thinkingText =
                cursor.getString(cursor.getColumnIndex(ChatDatabaseHelper.COLUMN_THINKING_TEXT))
            val thinkingFinishedTime =
                cursor.getLong(cursor.getColumnIndex(ChatDatabaseHelper.COLUMN_THINKING_FINISHED_TIME))
            chatDataItem.thinkingText = thinkingText
            chatDataItem.thinkingFinishedTime = thinkingFinishedTime
            chatDataItemList.add(chatDataItem)
        }
        cursor.close()
        db.close()

        return chatDataItemList
    }

    fun updateSessionName(sessionId: String, newName: String?) {
        val db = dbHelper.writableDatabase

        val values = ContentValues()
        values.put(ChatDatabaseHelper.COLUMN_SESSION_NAME, newName)

        db.update(
            ChatDatabaseHelper.TABLE_SESSION, values,
            ChatDatabaseHelper.COLUMN_SESSION_ID + "=?",
            arrayOf(sessionId)
        )
        db.close()
    }

    fun updateSessionModelId(sessionId: String, newModelId: String) {
        val db = dbHelper.writableDatabase

        val values = ContentValues()
        values.put(ChatDatabaseHelper.COLUMN_MODEL_ID, newModelId)

        db.update(
            ChatDatabaseHelper.TABLE_SESSION, values,
            ChatDatabaseHelper.COLUMN_SESSION_ID + "=?",
            arrayOf(sessionId)
        )
        db.close()
    }

    @get:SuppressLint("Range")
    val allSessions: MutableList<SessionItem>
        get() {
            val list: MutableList<SessionItem> =
                ArrayList()
            val db = dbHelper.readableDatabase

            val cursor = try {
                db.query(
                    ChatDatabaseHelper.TABLE_SESSION,
                    arrayOf(
                        ChatDatabaseHelper.COLUMN_SESSION_ID,
                        ChatDatabaseHelper.COLUMN_MODEL_ID,
                        ChatDatabaseHelper.COLUMN_SESSION_NAME,
                        ChatDatabaseHelper.COLUMN_LAST_CHAT_TIME,
                        ChatDatabaseHelper.COLUMN_SESSION_MODE
                    ),
                    null, null, null, null, ChatDatabaseHelper.COLUMN_LAST_CHAT_TIME + " DESC"
                )
            } catch (e: Exception) {
                // Fallback to query without lastChatTime column
                db.query(
                    ChatDatabaseHelper.TABLE_SESSION,
                    arrayOf(
                        ChatDatabaseHelper.COLUMN_SESSION_ID,
                        ChatDatabaseHelper.COLUMN_MODEL_ID,
                        ChatDatabaseHelper.COLUMN_SESSION_NAME
                    ),
                    null, null, null, null, ChatDatabaseHelper.COLUMN_SESSION_ID + " DESC"
                )
            }

            if (cursor != null) {
                while (cursor.moveToNext()) {
                    val sid =
                        cursor.getString(cursor.getColumnIndex(ChatDatabaseHelper.COLUMN_SESSION_ID))
                    val mid =
                        cursor.getString(cursor.getColumnIndex(ChatDatabaseHelper.COLUMN_MODEL_ID))
                    val name =
                        cursor.getString(cursor.getColumnIndex(ChatDatabaseHelper.COLUMN_SESSION_NAME))
                    val lastChatTime = try {
                        cursor.getLong(cursor.getColumnIndex(ChatDatabaseHelper.COLUMN_LAST_CHAT_TIME))
                    } catch (e: Exception) {
                        0L // Fallback for when column doesn't exist
                    }
                    val sessionMode = try {
                        cursor.getString(cursor.getColumnIndex(ChatDatabaseHelper.COLUMN_SESSION_MODE))
                    } catch (e: Exception) {
                        ChatDatabaseHelper.SESSION_MODE_NORMAL
                    }
                    list.add(SessionItem(sid, mid, name, lastChatTime, normalizeSessionMode(sessionMode)))
                }
                cursor.close()
            }
            db.close()
            return list
        }

    fun deleteAllChatData(sessionId: String) {
        val db = dbHelper.writableDatabase
        db.delete(
            ChatDatabaseHelper.TABLE_CHAT,
            ChatDatabaseHelper.COLUMN_SESSION_ID + "=?",
            arrayOf(sessionId)
        )
        db.close()
    }

    fun deleteSession(sessionId: String) {
        val db = dbHelper.writableDatabase
        try {
            db.delete(
                ChatDatabaseHelper.TABLE_CHAT,
                ChatDatabaseHelper.COLUMN_SESSION_ID + "=?",
                arrayOf(sessionId)
            )

            db.delete(
                ChatDatabaseHelper.TABLE_SESSION,
                ChatDatabaseHelper.COLUMN_SESSION_ID + "=?",
                arrayOf(sessionId)
            )
        } finally {
            db.close()
        }
    }

    /**
     * Delete all sessions for a given modelId.
     * Returns the list of deleted session IDs for resource cleanup.
     */
    fun deleteSessionsByModelId(modelId: String): List<String> {
        val deletedSessionIds = mutableListOf<String>()
        val db = dbHelper.writableDatabase
        try {
            // First get all session IDs for this model
            val cursor = db.query(
                ChatDatabaseHelper.TABLE_SESSION,
                arrayOf(ChatDatabaseHelper.COLUMN_SESSION_ID),
                ChatDatabaseHelper.COLUMN_MODEL_ID + "=?",
                arrayOf(modelId), null, null, null
            )
            while (cursor.moveToNext()) {
                deletedSessionIds.add(cursor.getString(0))
            }
            cursor.close()

            // Delete chat data for all sessions
            if (deletedSessionIds.isNotEmpty()) {
                val placeholders = deletedSessionIds.joinToString(",") { "?" }
                db.delete(
                    ChatDatabaseHelper.TABLE_CHAT,
                    ChatDatabaseHelper.COLUMN_SESSION_ID + " IN ($placeholders)",
                    deletedSessionIds.toTypedArray()
                )

                // Delete sessions
                db.delete(
                    ChatDatabaseHelper.TABLE_SESSION,
                    ChatDatabaseHelper.COLUMN_MODEL_ID + "=?",
                    arrayOf(modelId)
                )
            }
        } finally {
            db.close()
        }
        return deletedSessionIds
    }

    fun recordDownloadHistory(modelId: String, modelPath: String, modelType: String = "LLM") {
        val db = dbHelper.writableDatabase
        val values = ContentValues()
        values.put(ChatDatabaseHelper.COLUMN_DOWNLOAD_MODEL_ID, modelId)
        values.put(ChatDatabaseHelper.COLUMN_DOWNLOAD_TIME, System.currentTimeMillis())
        values.put(ChatDatabaseHelper.COLUMN_MODEL_PATH, modelPath)
        values.put(ChatDatabaseHelper.COLUMN_MODEL_TYPE, modelType)
        
        // Use INSERT OR REPLACE to handle duplicates
        db.replace(ChatDatabaseHelper.TABLE_DOWNLOAD_HISTORY, null, values)
        db.close()
    }

    @SuppressLint("Range")
    fun getDownloadTime(modelId: String): Long {
        val db = dbHelper.readableDatabase
        val cursor = db.query(
            ChatDatabaseHelper.TABLE_DOWNLOAD_HISTORY,
            arrayOf(ChatDatabaseHelper.COLUMN_DOWNLOAD_TIME),
            ChatDatabaseHelper.COLUMN_DOWNLOAD_MODEL_ID + "=?",
            arrayOf(modelId), null, null, null
        )
        
        var downloadTime = 0L
        if (cursor.moveToFirst()) {
            downloadTime = cursor.getLong(cursor.getColumnIndex(ChatDatabaseHelper.COLUMN_DOWNLOAD_TIME))
        }
        cursor.close()
        db.close()
        return downloadTime
    }

    @SuppressLint("Range")
    fun getDownloadModelType(modelId: String): String? {
        val db = dbHelper.readableDatabase
        val cursor = db.query(
            ChatDatabaseHelper.TABLE_DOWNLOAD_HISTORY,
            arrayOf(ChatDatabaseHelper.COLUMN_MODEL_TYPE),
            "${ChatDatabaseHelper.COLUMN_DOWNLOAD_MODEL_ID} = ?",
            arrayOf(modelId),
            null, null, null
        )
        var modelType: String? = null
        if (cursor.moveToFirst()) {
            modelType = cursor.getString(cursor.getColumnIndexOrThrow(ChatDatabaseHelper.COLUMN_MODEL_TYPE))
        }
        cursor.close()
        db.close()
        return modelType
    }

    @SuppressLint("Range")
    fun getLastChatTime(modelId: String): Long {
        val db = dbHelper.readableDatabase
        var lastChatTime = 0L
        var sessionId: String? = null
        try {
            val cursor = db.rawQuery(
                "SELECT MAX(${ChatDatabaseHelper.COLUMN_LAST_CHAT_TIME}) as lastChatTime, sessionId " +
                "FROM ${ChatDatabaseHelper.TABLE_SESSION} " +
                "WHERE ${ChatDatabaseHelper.COLUMN_MODEL_ID} = ?",
                arrayOf(modelId)
            )
            
            if (cursor.moveToFirst()) {
                lastChatTime = cursor.getLong(cursor.getColumnIndex(ChatDatabaseHelper.COLUMN_LAST_CHAT_TIME))
                sessionId = cursor.getString(cursor.getColumnIndex(ChatDatabaseHelper.COLUMN_SESSION_ID))
            }
            cursor.close()
        } catch (e: Exception) {
            Log.e(TAG, "getLastChatTime error: " + e.message, e )
        }
        if (lastChatTime <= 1000L && sessionId != null) {
            lastChatTime = sessionId.toLong()
            Log.d(TAG, "getLastChatTime not valid getLastChatTimeInChatData  sessionId: $sessionId modelId: $modelId result: $lastChatTime" )
        }
        db.close()
        return lastChatTime
    }


    @SuppressLint("Range")
    fun getSessionsForModel(modelId: String): List<SessionItem> {
        val list: MutableList<SessionItem> = ArrayList()
        val db = dbHelper.readableDatabase

        val cursor = db.query(
            ChatDatabaseHelper.TABLE_SESSION,
            arrayOf(
                ChatDatabaseHelper.COLUMN_SESSION_ID,
                ChatDatabaseHelper.COLUMN_MODEL_ID,
                ChatDatabaseHelper.COLUMN_SESSION_NAME,
                ChatDatabaseHelper.COLUMN_SESSION_MODE
            ),
            ChatDatabaseHelper.COLUMN_MODEL_ID + "=?",
            arrayOf(modelId), null, null, 
            ChatDatabaseHelper.COLUMN_SESSION_ID + " DESC"
        )

        while (cursor.moveToNext()) {
            val sid = cursor.getString(cursor.getColumnIndex(ChatDatabaseHelper.COLUMN_SESSION_ID))
            val mid = cursor.getString(cursor.getColumnIndex(ChatDatabaseHelper.COLUMN_MODEL_ID))
            val name = cursor.getString(cursor.getColumnIndex(ChatDatabaseHelper.COLUMN_SESSION_NAME))
            val sessionMode = try {
                cursor.getString(cursor.getColumnIndex(ChatDatabaseHelper.COLUMN_SESSION_MODE))
            } catch (e: Exception) {
                ChatDatabaseHelper.SESSION_MODE_NORMAL
            }
            list.add(SessionItem(sid, mid, name, sessionMode = normalizeSessionMode(sessionMode)))
        }
        cursor.close()
        db.close()
        return list
    }

    @SuppressLint("Range")
    fun getAllDownloadedModels(): List<DownloadedModelInfo> {
        val list: MutableList<DownloadedModelInfo> = ArrayList()
        val db = dbHelper.readableDatabase

        val cursor = try {
            db.rawQuery(
                "SELECT dh.${ChatDatabaseHelper.COLUMN_DOWNLOAD_MODEL_ID} as modelId, " +
                "dh.${ChatDatabaseHelper.COLUMN_DOWNLOAD_TIME} as downloadTime, " +
                "dh.${ChatDatabaseHelper.COLUMN_MODEL_PATH} as modelPath, " +
                "MAX(s.${ChatDatabaseHelper.COLUMN_LAST_CHAT_TIME}) as lastChatTime " +
                "FROM ${ChatDatabaseHelper.TABLE_DOWNLOAD_HISTORY} dh " +
                "LEFT JOIN ${ChatDatabaseHelper.TABLE_SESSION} s " +
                "ON dh.${ChatDatabaseHelper.COLUMN_DOWNLOAD_MODEL_ID} = s.${ChatDatabaseHelper.COLUMN_MODEL_ID} " +
                "GROUP BY dh.${ChatDatabaseHelper.COLUMN_DOWNLOAD_MODEL_ID}, dh.${ChatDatabaseHelper.COLUMN_DOWNLOAD_TIME}, dh.${ChatDatabaseHelper.COLUMN_MODEL_PATH}",
                null
            )
        } catch (e: Exception) {
            // Fallback to query without lastChatTime column
            db.rawQuery(
                "SELECT dh.${ChatDatabaseHelper.COLUMN_DOWNLOAD_MODEL_ID} as modelId, " +
                "dh.${ChatDatabaseHelper.COLUMN_DOWNLOAD_TIME} as downloadTime, " +
                "dh.${ChatDatabaseHelper.COLUMN_MODEL_PATH} as modelPath, " +
                "MAX(CAST(c.${ChatDatabaseHelper.COLUMN_TIME} AS INTEGER)) as lastChatTime " +
                "FROM ${ChatDatabaseHelper.TABLE_DOWNLOAD_HISTORY} dh " +
                "LEFT JOIN ${ChatDatabaseHelper.TABLE_SESSION} s " +
                "ON dh.${ChatDatabaseHelper.COLUMN_DOWNLOAD_MODEL_ID} = s.${ChatDatabaseHelper.COLUMN_MODEL_ID} " +
                "LEFT JOIN ${ChatDatabaseHelper.TABLE_CHAT} c " +
                "ON s.${ChatDatabaseHelper.COLUMN_SESSION_ID} = c.${ChatDatabaseHelper.COLUMN_SESSION_ID} " +
                "GROUP BY dh.${ChatDatabaseHelper.COLUMN_DOWNLOAD_MODEL_ID}, dh.${ChatDatabaseHelper.COLUMN_DOWNLOAD_TIME}, dh.${ChatDatabaseHelper.COLUMN_MODEL_PATH}",
                null
            )
        }

        while (cursor.moveToNext()) {
            val modelId = cursor.getString(cursor.getColumnIndex("modelId"))
            val downloadTime = cursor.getLong(cursor.getColumnIndex("downloadTime"))
            val modelPath = cursor.getString(cursor.getColumnIndex("modelPath"))
            var lastChatTime = cursor.getLong(cursor.getColumnIndex("lastChatTime"))
            
            // If lastChatTime is 0, try to fix it by querying from TABLE_CHAT
            if (lastChatTime == 0L) {
                lastChatTime = fixMissingLastChatTimeForModel(modelId)
            }
            
            list.add(DownloadedModelInfo(modelId, downloadTime, modelPath, lastChatTime))
        }
        cursor.close()
        db.close()
        return list
    }

    data class DownloadedModelInfo(
        val modelId: String,
        val downloadTime: Long,
        val modelPath: String,
        val lastChatTime: Long
    ) {
        fun getDisplayTime(): Long {
            return if (lastChatTime > 0) lastChatTime else downloadTime
        }
        
        fun hasChated(): Boolean {
            return lastChatTime > 0
        }
    }
    
    @SuppressLint("Range")
    fun fixAllMissingLastChatTimes() {
        val db = dbHelper.readableDatabase
        try {
            // Find all sessions with lastChatTime = 0 or NULL
            val cursor = db.query(
                ChatDatabaseHelper.TABLE_SESSION,
                arrayOf(ChatDatabaseHelper.COLUMN_SESSION_ID),
                "${ChatDatabaseHelper.COLUMN_LAST_CHAT_TIME} IS NULL OR ${ChatDatabaseHelper.COLUMN_LAST_CHAT_TIME} = 0",
                null, null, null, null
            )
            
            while (cursor.moveToNext()) {
                val sessionId = cursor.getString(cursor.getColumnIndex(ChatDatabaseHelper.COLUMN_SESSION_ID))
                fixMissingLastChatTime(sessionId)
            }
            cursor.close()
        } finally {
            db.close()
        }
    }
    
    private fun ensureLastChatTimeColumn() {
        val db = dbHelper.writableDatabase
        try {
            // Try to query the column to see if it exists
            db.rawQuery("SELECT ${ChatDatabaseHelper.COLUMN_LAST_CHAT_TIME} FROM ${ChatDatabaseHelper.TABLE_SESSION} LIMIT 1", null)?.close()
        } catch (e: Exception) {
            try {
                // Column doesn't exist, add it
                db.execSQL("ALTER TABLE ${ChatDatabaseHelper.TABLE_SESSION} ADD COLUMN ${ChatDatabaseHelper.COLUMN_LAST_CHAT_TIME} INTEGER DEFAULT 0")
            } catch (e2: Exception) {
                // Ignore if already exists
            }
        } finally {
            db.close()
        }
    }

    private fun ensureSessionModeColumn() {
        val db = dbHelper.writableDatabase
        try {
            db.rawQuery("SELECT ${ChatDatabaseHelper.COLUMN_SESSION_MODE} FROM ${ChatDatabaseHelper.TABLE_SESSION} LIMIT 1", null)?.close()
        } catch (e: Exception) {
            try {
                db.execSQL(
                    "ALTER TABLE ${ChatDatabaseHelper.TABLE_SESSION} " +
                        "ADD COLUMN ${ChatDatabaseHelper.COLUMN_SESSION_MODE} TEXT DEFAULT '${ChatDatabaseHelper.SESSION_MODE_NORMAL}'"
                )
            } catch (e2: Exception) {
                // Ignore if already exists
            }
        } finally {
            db.close()
        }
    }

    private fun ensureAgentTables() {
        val db = dbHelper.writableDatabase
        try {
            db.execSQL(
                "CREATE TABLE IF NOT EXISTS ${ChatDatabaseHelper.TABLE_AGENT_MEMORY} (" +
                    "${ChatDatabaseHelper.COLUMN_AGENT_MEMORY_ID} INTEGER PRIMARY KEY AUTOINCREMENT, " +
                    "${ChatDatabaseHelper.COLUMN_AGENT_MEMORY_CATEGORY} TEXT, " +
                    "${ChatDatabaseHelper.COLUMN_AGENT_MEMORY_CONTENT} TEXT, " +
                    "${ChatDatabaseHelper.COLUMN_AGENT_MEMORY_SOURCE} TEXT DEFAULT 'agent', " +
                    "${ChatDatabaseHelper.COLUMN_AGENT_MEMORY_UPDATED_AT} INTEGER)"
            )
            db.execSQL(
                "CREATE TABLE IF NOT EXISTS ${ChatDatabaseHelper.TABLE_AGENT_SKILL} (" +
                    "${ChatDatabaseHelper.COLUMN_AGENT_SKILL_ID} INTEGER PRIMARY KEY AUTOINCREMENT, " +
                    "${ChatDatabaseHelper.COLUMN_AGENT_SKILL_NAME} TEXT UNIQUE, " +
                    "${ChatDatabaseHelper.COLUMN_AGENT_SKILL_DESCRIPTION} TEXT, " +
                    "${ChatDatabaseHelper.COLUMN_AGENT_SKILL_TRIGGER_KEYWORDS} TEXT, " +
                    "${ChatDatabaseHelper.COLUMN_AGENT_SKILL_ACTION_TEMPLATE} TEXT, " +
                    "${ChatDatabaseHelper.COLUMN_AGENT_SKILL_ENABLED} INTEGER DEFAULT 1, " +
                    "${ChatDatabaseHelper.COLUMN_AGENT_SKILL_CREATED_AT} INTEGER)"
            )
        } finally {
            db.close()
        }
    }

    private fun normalizeSessionMode(sessionMode: String?): String {
        return if (sessionMode == ChatDatabaseHelper.SESSION_MODE_AGENT) {
            ChatDatabaseHelper.SESSION_MODE_AGENT
        } else {
            ChatDatabaseHelper.SESSION_MODE_NORMAL
        }
    }

    @SuppressLint("Range")
    private fun fixMissingLastChatTime(sessionId: String): Long {
        val db = dbHelper.writableDatabase
        try {
            // Query the latest chat time from TABLE_CHAT for this session
            val cursor = db.rawQuery(
                "SELECT MAX(CAST(${ChatDatabaseHelper.COLUMN_TIME} AS INTEGER)) as latestChatTime " +
                "FROM ${ChatDatabaseHelper.TABLE_CHAT} " +
                "WHERE ${ChatDatabaseHelper.COLUMN_SESSION_ID} = ?",
                arrayOf(sessionId)
            )
            
            var latestChatTime = 0L
            if (cursor.moveToFirst()) {
                latestChatTime = cursor.getLong(cursor.getColumnIndex("latestChatTime"))
            }
            cursor.close()
            
            // Update the session's lastChatTime if we found a valid time
            if (latestChatTime > 0) {
                val values = ContentValues()
                values.put(ChatDatabaseHelper.COLUMN_LAST_CHAT_TIME, latestChatTime)
                db.update(
                    ChatDatabaseHelper.TABLE_SESSION,
                    values,
                    ChatDatabaseHelper.COLUMN_SESSION_ID + "=?",
                    arrayOf(sessionId)
                )
            }
            
            return latestChatTime
        } catch (e: Exception) {
            return 0L
        } finally {
            db.close()
        }
    }

    @SuppressLint("Range")
    private fun fixMissingLastChatTimeForModel(modelId: String): Long {
        Log.d(TAG, "fixMissingLastChatTimeForModel: $modelId")
        val db = dbHelper.writableDatabase
        try {
            // Query all sessions for this model that have lastChatTime = 0
            val sessionCursor = db.query(
                ChatDatabaseHelper.TABLE_SESSION,
                arrayOf(ChatDatabaseHelper.COLUMN_SESSION_ID),
                "${ChatDatabaseHelper.COLUMN_MODEL_ID} = ? AND (${ChatDatabaseHelper.COLUMN_LAST_CHAT_TIME} IS NULL OR ${ChatDatabaseHelper.COLUMN_LAST_CHAT_TIME} = 0)",
                arrayOf(modelId), null, null, null
            )
            
            var maxChatTime = 0L
            while (sessionCursor.moveToNext()) {
                val sessionId = sessionCursor.getString(sessionCursor.getColumnIndex(ChatDatabaseHelper.COLUMN_SESSION_ID))
                val chatTime = fixMissingLastChatTime(sessionId)
                if (chatTime > maxChatTime) {
                    maxChatTime = chatTime
                }
            }
            sessionCursor.close()
            Log.d(TAG, "fixMissingLastChatTimeForModel: $modelId maxChatTime: ${maxChatTime}")
            return maxChatTime
        } catch (e: Exception) {
            Log.e(TAG, "fixMissingLastChatTimeForModel: $modelId hasException", e)
            return 0L
        } finally {
            db.close()
        }
    }

    companion object {
        private var sInstance: ChatDataManager? = null
        private const val TAG = "ChatDataManager"
        private const val MEMORY_CONTENT_MAX_CHARS = 168
        private const val SKILL_ACTION_MAX_CHARS = 350

        @JvmStatic
        fun getInstance(context: Context): ChatDataManager {
            synchronized(ChatDataManager::class.java) {
                if (sInstance == null) {
                    sInstance = ChatDataManager(context.applicationContext)
                }
            }
            return sInstance!!
        }
    }
}
