// Created by ruoyi.sjd on 2025/01/05.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.chat;

import android.content.Context;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;

public class ChatDatabaseHelper extends SQLiteOpenHelper {

    private static final String DB_NAME = "chat.db";
    private static final int DB_VERSION = 1;

    // 会话表
    public static final String TABLE_SESSION = "Session";
    public static final String COLUMN_SESSION_ID = "sessionId";
    public static final String COLUMN_MODEL_ID = "modelId";
    public static final String COLUMN_SESSION_NAME = "name";

    public static final String TABLE_CHAT = "ChatData";
    public static final String COLUMN_ID = "_id";
    public static final String COLUMN_TIME = "time";
    public static final String COLUMN_TEXT = "text";
    public static final String COLUMN_TYPE = "type";
    public static final String COLUMN_IMAGE_URI = "imageUri";

    public static final String COLUMN_AUDIO_URI = "audioUri";
    public static final String COLUMN_AUDIO_DURATION = "audioDuration";


    private static final String CREATE_TABLE_SESSION = "CREATE TABLE IF NOT EXISTS " +
            TABLE_SESSION + " (" +
            COLUMN_SESSION_ID + " TEXT PRIMARY KEY, " +
            COLUMN_MODEL_ID + " TEXT," +
            COLUMN_SESSION_NAME + " TEXT)";

    private static final String CREATE_TABLE_CHAT = "CREATE TABLE IF NOT EXISTS " +
            TABLE_CHAT + " (" +
            COLUMN_ID + " INTEGER PRIMARY KEY AUTOINCREMENT, " +
            COLUMN_SESSION_ID + " TEXT, " +
            COLUMN_TIME + " TEXT, " +
            COLUMN_TYPE + " INTEGER, " +
            COLUMN_TEXT + " TEXT, " +
            COLUMN_IMAGE_URI + " TEXT," +
            COLUMN_AUDIO_URI + " TEXT," +
            COLUMN_AUDIO_DURATION + " REAL)";

    public ChatDatabaseHelper(Context context) {
        super(context, DB_NAME, null, DB_VERSION);
    }

    @Override
    public void onCreate(SQLiteDatabase db) {
        db.execSQL(CREATE_TABLE_SESSION);
        db.execSQL(CREATE_TABLE_CHAT);
    }

    @Override
    public void onUpgrade(SQLiteDatabase db, int oldVersion, int newVersion) {
        db.execSQL("DROP TABLE IF EXISTS " + TABLE_SESSION);
        db.execSQL("DROP TABLE IF EXISTS " + TABLE_CHAT);
        onCreate(db);
    }
}
