// Created by ruoyi.sjd on 2025/01/05.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.chat.model
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper

class ChatDatabaseHelper(context: Context?) :
    SQLiteOpenHelper(context, DB_NAME, null, DB_VERSION) {
    override fun onCreate(db: SQLiteDatabase) {
        db.execSQL(CREATE_TABLE_SESSION)
        db.execSQL(CREATE_TABLE_CHAT)
    }

    override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
        if (oldVersion < 2) {
            db.execSQL("ALTER TABLE $TABLE_CHAT ADD COLUMN $COLUMN_RESERVE1 TEXT")
            db.execSQL("ALTER TABLE $TABLE_CHAT ADD COLUMN $COLUMN_RESERVE2 TEXT")
            db.execSQL("ALTER TABLE $TABLE_CHAT ADD COLUMN $COLUMN_DISPLAY_TEXT TEXT")
        }
        if (oldVersion < 4) {
            db.execSQL("ALTER TABLE $TABLE_CHAT ADD COLUMN $COLUMN_THINKING_TEXT TEXT")
            db.execSQL("ALTER TABLE $TABLE_CHAT ADD COLUMN $COLUMN_THINKING_FINISHED_TIME INTEGER DEFAULT 0")
        }
    }

    companion object {
        private const val DB_NAME = "chat.db"
        private const val DB_VERSION = 4
        const val TABLE_SESSION: String = "Session"
        const val COLUMN_SESSION_ID: String = "sessionId"
        const val COLUMN_MODEL_ID: String = "modelId"
        const val COLUMN_SESSION_NAME: String = "name"

        const val TABLE_CHAT: String = "ChatData"
        const val COLUMN_ID: String = "_id"
        const val COLUMN_TIME: String = "time"
        const val COLUMN_TEXT: String = "text"
        const val COLUMN_TYPE: String = "type"
        const val COLUMN_IMAGE_URI: String = "imageUri"

        const val COLUMN_AUDIO_URI: String = "audioUri"
        const val COLUMN_AUDIO_DURATION: String = "audioDuration"

        const val COLUMN_RESERVE1: String = "reserve1"
        const val COLUMN_RESERVE2: String = "reserve2"
        const val COLUMN_DISPLAY_TEXT: String = "displayText"
        const val COLUMN_THINKING_TEXT: String = "thinkingText"
        const val COLUMN_THINKING_FINISHED_TIME: String = "thinkingFinishedTime"

        private const val CREATE_TABLE_SESSION = "CREATE TABLE IF NOT EXISTS " +
                TABLE_SESSION + " (" +
                COLUMN_SESSION_ID + " TEXT PRIMARY KEY, " +
                COLUMN_MODEL_ID + " TEXT," +
                COLUMN_SESSION_NAME + " TEXT)"

        private const val CREATE_TABLE_CHAT = "CREATE TABLE IF NOT EXISTS " +
                TABLE_CHAT + " (" +
                COLUMN_ID + " INTEGER PRIMARY KEY AUTOINCREMENT, " +
                COLUMN_SESSION_ID + " TEXT, " +
                COLUMN_TIME + " TEXT, " +
                COLUMN_TYPE + " INTEGER, " +
                COLUMN_TEXT + " TEXT, " +
                COLUMN_IMAGE_URI + " TEXT," +
                COLUMN_AUDIO_URI + " TEXT," +
                COLUMN_AUDIO_DURATION + " REAL," +
                COLUMN_DISPLAY_TEXT + " TEXT," +
                COLUMN_RESERVE1 + " TEXT, " +
                COLUMN_RESERVE2 + " TEXT," +
                COLUMN_THINKING_TEXT + " TEXT, " +
                COLUMN_THINKING_FINISHED_TIME + " INTEGER)"
    }
}
