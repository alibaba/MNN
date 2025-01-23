// Created by ruoyi.sjd on 2025/01/05.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.

package com.alibaba.mnnllm.android.chat;


import android.annotation.SuppressLint;
import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.net.Uri;

import java.util.ArrayList;
import java.util.List;

public class ChatDataManager {

    private ChatDatabaseHelper dbHelper;
    private static ChatDataManager sInstance;

    private ChatDataManager(Context context) {
        dbHelper = new ChatDatabaseHelper(context);
    }

    public static ChatDataManager getInstance(Context context) {
        synchronized (ChatDataManager.class) {
            if (sInstance == null) {
                sInstance = new ChatDataManager(context.getApplicationContext());
            }
        }
        return sInstance;
    }

    public void addOrUpdateSession(String sessionId, String modelId) {
        SQLiteDatabase db = dbHelper.getWritableDatabase();

        Cursor cursor = db.query(ChatDatabaseHelper.TABLE_SESSION,
                new String[]{ChatDatabaseHelper.COLUMN_SESSION_ID},
                ChatDatabaseHelper.COLUMN_SESSION_ID + "=?",
                new String[]{sessionId}, null, null, null);

        boolean exists = cursor.moveToFirst();
        cursor.close();

        ContentValues values = new ContentValues();
        values.put(ChatDatabaseHelper.COLUMN_SESSION_ID, sessionId);
        values.put(ChatDatabaseHelper.COLUMN_MODEL_ID, modelId);

        if (exists) {
            db.update(ChatDatabaseHelper.TABLE_SESSION, values,
                    ChatDatabaseHelper.COLUMN_SESSION_ID + "=?",
                    new String[]{sessionId});
        } else {
            db.insert(ChatDatabaseHelper.TABLE_SESSION, null, values);
        }
        db.close();
    }

    public void addChatData(String sessionId, ChatDataItem chatDataItem) {
        SQLiteDatabase db = dbHelper.getWritableDatabase();
        ContentValues values = new ContentValues();
        values.put(ChatDatabaseHelper.COLUMN_SESSION_ID, sessionId);
        values.put(ChatDatabaseHelper.COLUMN_TIME, chatDataItem.getTime());
        values.put(ChatDatabaseHelper.COLUMN_TEXT, chatDataItem.getText());
        values.put(ChatDatabaseHelper.COLUMN_TYPE, chatDataItem.getType());
        if (chatDataItem.getImageUri() != null) {
            values.put(ChatDatabaseHelper.COLUMN_IMAGE_URI, chatDataItem.getImageUri().toString());
        } else {
            values.put(ChatDatabaseHelper.COLUMN_IMAGE_URI, (String) null);
        }
        if (chatDataItem.getAudioUri() != null) {
            values.put(ChatDatabaseHelper.COLUMN_AUDIO_URI, chatDataItem.getAudioUri().toString());
        } else {
            values.put(ChatDatabaseHelper.COLUMN_AUDIO_URI, (String) null);
        }
        values.put(ChatDatabaseHelper.COLUMN_AUDIO_DURATION, chatDataItem.getAudioDuration());
        db.insert(ChatDatabaseHelper.TABLE_CHAT, null, values);
        db.close();
    }

    @SuppressLint("Range")
    public List<ChatDataItem> getChatDataBySession(String sessionId) {
        List<ChatDataItem> chatDataItemList = new ArrayList<>();
        SQLiteDatabase db = dbHelper.getReadableDatabase();

        Cursor cursor = db.query(ChatDatabaseHelper.TABLE_CHAT,
                null,
                ChatDatabaseHelper.COLUMN_SESSION_ID + "=?",
                new String[]{sessionId}, null, null, null);

        if (cursor != null) {
            while (cursor.moveToNext()) {
                 String time = cursor.getString(cursor.getColumnIndex(ChatDatabaseHelper.COLUMN_TIME));
                int type = cursor.getInt(cursor.getColumnIndex(ChatDatabaseHelper.COLUMN_TYPE));
                String text = cursor.getString(cursor.getColumnIndex(ChatDatabaseHelper.COLUMN_TEXT));
                String imageUriStr = cursor.getString(cursor.getColumnIndex(ChatDatabaseHelper.COLUMN_IMAGE_URI));
                String audioUriStr = cursor.getString(cursor.getColumnIndex(ChatDatabaseHelper.COLUMN_AUDIO_URI));
                float audioDuration = cursor.getFloat(cursor.getColumnIndex(ChatDatabaseHelper.COLUMN_AUDIO_DURATION));
                ChatDataItem chatDataItem = new ChatDataItem(time, type, text);
                if (imageUriStr != null) {
                    chatDataItem.setImageUri(Uri.parse(imageUriStr));
                }
                if (audioUriStr != null) {
                    chatDataItem.setAudioUri(Uri.parse(audioUriStr));
                    chatDataItem.setAudioDuration(audioDuration);
                }
                chatDataItemList.add(chatDataItem);
            }
            cursor.close();
        }
        db.close();

        return chatDataItemList;
    }

    public void updateSessionName(String sessionId, String newName) {
        SQLiteDatabase db = dbHelper.getWritableDatabase();

        ContentValues values = new ContentValues();
        values.put(ChatDatabaseHelper.COLUMN_SESSION_NAME, newName);

        db.update(ChatDatabaseHelper.TABLE_SESSION, values,
                ChatDatabaseHelper.COLUMN_SESSION_ID + "=?",
                new String[]{sessionId});
        db.close();
    }

    @SuppressLint("Range")
    public List<SessionItem> getAllSessions() {
        List<SessionItem> list = new ArrayList<>();
        SQLiteDatabase db = dbHelper.getReadableDatabase();

        Cursor cursor = db.query(ChatDatabaseHelper.TABLE_SESSION,
                new String[]{ChatDatabaseHelper.COLUMN_SESSION_ID,
                        ChatDatabaseHelper.COLUMN_MODEL_ID,
                        ChatDatabaseHelper.COLUMN_SESSION_NAME},
                null, null, null, null, ChatDatabaseHelper.COLUMN_SESSION_ID + " DESC");

        if (cursor != null) {
            while (cursor.moveToNext()) {
                 String sid = cursor.getString(cursor.getColumnIndex(ChatDatabaseHelper.COLUMN_SESSION_ID));
                String mid = cursor.getString(cursor.getColumnIndex(ChatDatabaseHelper.COLUMN_MODEL_ID));
                String name = cursor.getString(cursor.getColumnIndex(ChatDatabaseHelper.COLUMN_SESSION_NAME));
                list.add(new SessionItem(sid, mid, name));
            }
            cursor.close();
        }
        db.close();
        return list;
    }

    public void deleteAllChatData(String sessionId) {
        SQLiteDatabase db = dbHelper.getWritableDatabase();
        db.delete(ChatDatabaseHelper.TABLE_CHAT,
                ChatDatabaseHelper.COLUMN_SESSION_ID + "=?",
                new String[]{sessionId});
        db.close();
    }

    public void deleteSession(String sessionId) {
        SQLiteDatabase db = dbHelper.getWritableDatabase();
        try {
            db.delete(ChatDatabaseHelper.TABLE_CHAT,
                    ChatDatabaseHelper.COLUMN_SESSION_ID + "=?",
                    new String[]{sessionId});

            db.delete(ChatDatabaseHelper.TABLE_SESSION,
                    ChatDatabaseHelper.COLUMN_SESSION_ID + "=?",
                    new String[]{sessionId});
        } finally {
            db.close();
        }
    }
}
