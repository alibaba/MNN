// Created by ruoyi.sjd on 2025/01/05.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
package com.alibaba.mnnllm.android.chat

import android.annotation.SuppressLint
import android.content.ContentValues
import android.content.Context
import android.net.Uri
import android.text.TextUtils


class ChatDataManager private constructor(context: Context) {
    private val dbHelper = ChatDatabaseHelper(context)

    fun addOrUpdateSession(sessionId: String, modelId: String?) {
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

        if (exists) {
            db.update(
                ChatDatabaseHelper.TABLE_SESSION, values,
                ChatDatabaseHelper.COLUMN_SESSION_ID + "=?",
                arrayOf(sessionId)
            )
        } else {
            db.insert(ChatDatabaseHelper.TABLE_SESSION, null, values)
        }
        db.close()
    }

    fun addChatData(sessionId: String?, chatDataItem: ChatDataItem) {
        val db = dbHelper.writableDatabase
        val values = ContentValues()
        values.put(ChatDatabaseHelper.COLUMN_SESSION_ID, sessionId)
        values.put(ChatDatabaseHelper.COLUMN_TIME, chatDataItem.time)
        values.put(ChatDatabaseHelper.COLUMN_TEXT, chatDataItem.text)
        values.put(ChatDatabaseHelper.COLUMN_TYPE, chatDataItem.type)
        if (chatDataItem.imageUri != null) {
            values.put(ChatDatabaseHelper.COLUMN_IMAGE_URI, chatDataItem.imageUri.toString())
        } else {
            values.put(ChatDatabaseHelper.COLUMN_IMAGE_URI, null as String?)
        }
        if (chatDataItem.audioUri != null) {
            values.put(ChatDatabaseHelper.COLUMN_AUDIO_URI, chatDataItem.audioUri.toString())
        } else {
            values.put(ChatDatabaseHelper.COLUMN_AUDIO_URI, null as String?)
        }
        values.put(ChatDatabaseHelper.COLUMN_AUDIO_DURATION, chatDataItem.audioDuration)
        values.put(ChatDatabaseHelper.COLUMN_DISPLAY_TEXT, chatDataItem.displayText)
        db.insert(ChatDatabaseHelper.TABLE_CHAT, null, values)
        db.close()
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

        if (cursor != null) {
            while (cursor.moveToNext()) {
                val time = cursor.getString(cursor.getColumnIndex(ChatDatabaseHelper.COLUMN_TIME))
                val type = cursor.getInt(cursor.getColumnIndex(ChatDatabaseHelper.COLUMN_TYPE))
                val text = cursor.getString(cursor.getColumnIndex(ChatDatabaseHelper.COLUMN_TEXT))
                val imageUriStr =
                    cursor.getString(cursor.getColumnIndex(ChatDatabaseHelper.COLUMN_IMAGE_URI))
                val audioUriStr =
                    cursor.getString(cursor.getColumnIndex(ChatDatabaseHelper.COLUMN_AUDIO_URI))
                val audioDuration =
                    cursor.getFloat(cursor.getColumnIndex(ChatDatabaseHelper.COLUMN_AUDIO_DURATION))
                val displayText =
                    cursor.getString(cursor.getColumnIndex(ChatDatabaseHelper.COLUMN_DISPLAY_TEXT))
                val chatDataItem = ChatDataItem(time, type, text)
                if (imageUriStr != null) {
                    chatDataItem.imageUri = Uri.parse(imageUriStr)
                }
                if (!TextUtils.isEmpty(displayText)) {
                    chatDataItem.displayText = displayText
                }
                if (audioUriStr != null) {
                    chatDataItem.audioUri = Uri.parse(audioUriStr)
                    chatDataItem.audioDuration = audioDuration
                }
                chatDataItemList.add(chatDataItem)
            }
            cursor.close()
        }
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

    @get:SuppressLint("Range")
    val allSessions: MutableList<SessionItem>
        get() {
            val list: MutableList<SessionItem> =
                ArrayList()
            val db = dbHelper.readableDatabase

            val cursor = db.query(
                ChatDatabaseHelper.TABLE_SESSION,
                arrayOf(
                    ChatDatabaseHelper.COLUMN_SESSION_ID,
                    ChatDatabaseHelper.COLUMN_MODEL_ID,
                    ChatDatabaseHelper.COLUMN_SESSION_NAME
                ),
                null, null, null, null, ChatDatabaseHelper.COLUMN_SESSION_ID + " DESC"
            )

            if (cursor != null) {
                while (cursor.moveToNext()) {
                    val sid =
                        cursor.getString(cursor.getColumnIndex(ChatDatabaseHelper.COLUMN_SESSION_ID))
                    val mid =
                        cursor.getString(cursor.getColumnIndex(ChatDatabaseHelper.COLUMN_MODEL_ID))
                    val name =
                        cursor.getString(cursor.getColumnIndex(ChatDatabaseHelper.COLUMN_SESSION_NAME))
                    list.add(SessionItem(sid, mid, name))
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

    companion object {
        private var sInstance: ChatDataManager? = null

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
