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
        db.execSQL(CREATE_TABLE_DOWNLOAD_HISTORY)
        db.execSQL(CREATE_TABLE_AGENT_MEMORY)
        db.execSQL(CREATE_TABLE_AGENT_SKILL)
    }

    override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
        if (oldVersion < 2) {
            try {
                db.execSQL("ALTER TABLE $TABLE_CHAT ADD COLUMN $COLUMN_RESERVE1 TEXT")
                db.execSQL("ALTER TABLE $TABLE_CHAT ADD COLUMN $COLUMN_RESERVE2 TEXT")
                db.execSQL("ALTER TABLE $TABLE_CHAT ADD COLUMN $COLUMN_DISPLAY_TEXT TEXT")
            } catch (e: Exception) {
                // Column might already exist
            }
        }
        if (oldVersion < 4) {
            try {
                db.execSQL("ALTER TABLE $TABLE_CHAT ADD COLUMN $COLUMN_THINKING_TEXT TEXT")
                db.execSQL("ALTER TABLE $TABLE_CHAT ADD COLUMN $COLUMN_THINKING_FINISHED_TIME INTEGER DEFAULT 0")
            } catch (e: Exception) {
                // Column might already exist
            }
        }
        if (oldVersion < 5) {
            try {
                db.execSQL(CREATE_TABLE_DOWNLOAD_HISTORY)
            } catch (e: Exception) {
                // Table might already exist
            }
        }
        if (oldVersion < 6) {
            try {
                db.execSQL("ALTER TABLE $TABLE_SESSION ADD COLUMN $COLUMN_LAST_CHAT_TIME INTEGER DEFAULT 0")
            } catch (e: Exception) {
                // Column might already exist, ignore
            }
        }
        if (oldVersion < 7) {
            try {
                db.execSQL("ALTER TABLE $TABLE_DOWNLOAD_HISTORY ADD COLUMN $COLUMN_MODEL_TYPE TEXT DEFAULT 'LLM'")
            } catch (e: Exception) {
                // Column might already exist, ignore
            }
        }
        if (oldVersion < 8) {
            try {
                db.execSQL("ALTER TABLE $TABLE_CHAT ADD COLUMN $COLUMN_VIDEO_URI TEXT")
            } catch (e: Exception) {
                // Column might already exist, ignore
            }
        }
        if (oldVersion < 9) {
            try {
                db.execSQL("ALTER TABLE $TABLE_SESSION ADD COLUMN $COLUMN_SESSION_MODE TEXT DEFAULT '$SESSION_MODE_NORMAL'")
            } catch (e: Exception) {
                // Column might already exist, ignore
            }
        }
        if (oldVersion < 10) {
            try {
                db.execSQL(CREATE_TABLE_AGENT_MEMORY)
                db.execSQL(CREATE_TABLE_AGENT_SKILL)
            } catch (e: Exception) {
                // Tables might already exist, ignore
            }
        }
    }

    companion object {
        private const val DB_NAME = "chat.db"
        private const val DB_VERSION = 10
        const val TABLE_SESSION: String = "Session"
        const val COLUMN_SESSION_ID: String = "sessionId"
        const val COLUMN_MODEL_ID: String = "modelId"
        const val COLUMN_SESSION_NAME: String = "name"
        const val COLUMN_LAST_CHAT_TIME: String = "lastChatTime"
        const val COLUMN_SESSION_MODE: String = "sessionMode"
        const val SESSION_MODE_NORMAL: String = "normal"
        const val SESSION_MODE_AGENT: String = "agent"

        const val TABLE_CHAT: String = "ChatData"
        const val COLUMN_ID: String = "_id"
        const val COLUMN_TIME: String = "time"
        const val COLUMN_TEXT: String = "text"
        const val COLUMN_TYPE: String = "type"
        const val COLUMN_IMAGE_URI: String = "imageUri"

        const val COLUMN_AUDIO_URI: String = "audioUri"
        const val COLUMN_AUDIO_DURATION: String = "audioDuration"
        const val COLUMN_VIDEO_URI: String = "videoUri"

        const val COLUMN_RESERVE1: String = "reserve1"
        const val COLUMN_RESERVE2: String = "reserve2"
        const val COLUMN_DISPLAY_TEXT: String = "displayText"
        const val COLUMN_THINKING_TEXT: String = "thinkingText"
        const val COLUMN_THINKING_FINISHED_TIME: String = "thinkingFinishedTime"

        // Download history table
        const val TABLE_DOWNLOAD_HISTORY: String = "DownloadHistory"
        const val COLUMN_DOWNLOAD_MODEL_ID: String = "modelId"
        const val COLUMN_DOWNLOAD_TIME: String = "downloadTime"
        const val COLUMN_MODEL_PATH: String = "modelPath"
        const val COLUMN_MODEL_TYPE: String = "modelType"

        const val TABLE_AGENT_MEMORY: String = "AgentMemory"
        const val COLUMN_AGENT_MEMORY_ID: String = "_id"
        const val COLUMN_AGENT_MEMORY_CATEGORY: String = "category"
        const val COLUMN_AGENT_MEMORY_CONTENT: String = "content"
        const val COLUMN_AGENT_MEMORY_SOURCE: String = "source"
        const val COLUMN_AGENT_MEMORY_UPDATED_AT: String = "updatedAt"

        const val TABLE_AGENT_SKILL: String = "AgentSkill"
        const val COLUMN_AGENT_SKILL_ID: String = "_id"
        const val COLUMN_AGENT_SKILL_NAME: String = "name"
        const val COLUMN_AGENT_SKILL_DESCRIPTION: String = "description"
        const val COLUMN_AGENT_SKILL_TRIGGER_KEYWORDS: String = "triggerKeywords"
        const val COLUMN_AGENT_SKILL_ACTION_TEMPLATE: String = "actionTemplate"
        const val COLUMN_AGENT_SKILL_ENABLED: String = "enabled"
        const val COLUMN_AGENT_SKILL_CREATED_AT: String = "createdAt"

        private const val CREATE_TABLE_SESSION = "CREATE TABLE IF NOT EXISTS " +
                TABLE_SESSION + " (" +
                COLUMN_SESSION_ID + " TEXT PRIMARY KEY, " +
                COLUMN_MODEL_ID + " TEXT," +
                COLUMN_SESSION_NAME + " TEXT," +
                COLUMN_LAST_CHAT_TIME + " INTEGER DEFAULT 0," +
                COLUMN_SESSION_MODE + " TEXT DEFAULT '" + SESSION_MODE_NORMAL + "')"

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
                COLUMN_VIDEO_URI + " TEXT," +
                COLUMN_DISPLAY_TEXT + " TEXT," +
                COLUMN_RESERVE1 + " TEXT, " +
                COLUMN_RESERVE2 + " TEXT," +
                COLUMN_THINKING_TEXT + " TEXT, " +
                COLUMN_THINKING_FINISHED_TIME + " INTEGER)"

        private const val CREATE_TABLE_DOWNLOAD_HISTORY = "CREATE TABLE IF NOT EXISTS " +
                TABLE_DOWNLOAD_HISTORY + " (" +
                COLUMN_DOWNLOAD_MODEL_ID + " TEXT PRIMARY KEY, " +
                COLUMN_DOWNLOAD_TIME + " INTEGER, " +
                COLUMN_MODEL_PATH + " TEXT, " +
                COLUMN_MODEL_TYPE + " TEXT DEFAULT 'LLM')"

        private const val CREATE_TABLE_AGENT_MEMORY = "CREATE TABLE IF NOT EXISTS " +
                TABLE_AGENT_MEMORY + " (" +
                COLUMN_AGENT_MEMORY_ID + " INTEGER PRIMARY KEY AUTOINCREMENT, " +
                COLUMN_AGENT_MEMORY_CATEGORY + " TEXT, " +
                COLUMN_AGENT_MEMORY_CONTENT + " TEXT, " +
                COLUMN_AGENT_MEMORY_SOURCE + " TEXT DEFAULT 'agent', " +
                COLUMN_AGENT_MEMORY_UPDATED_AT + " INTEGER)"

        private const val CREATE_TABLE_AGENT_SKILL = "CREATE TABLE IF NOT EXISTS " +
                TABLE_AGENT_SKILL + " (" +
                COLUMN_AGENT_SKILL_ID + " INTEGER PRIMARY KEY AUTOINCREMENT, " +
                COLUMN_AGENT_SKILL_NAME + " TEXT UNIQUE, " +
                COLUMN_AGENT_SKILL_DESCRIPTION + " TEXT, " +
                COLUMN_AGENT_SKILL_TRIGGER_KEYWORDS + " TEXT, " +
                COLUMN_AGENT_SKILL_ACTION_TEMPLATE + " TEXT, " +
                COLUMN_AGENT_SKILL_ENABLED + " INTEGER DEFAULT 1, " +
                COLUMN_AGENT_SKILL_CREATED_AT + " INTEGER)"
    }
}
