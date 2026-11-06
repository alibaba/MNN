package com.alibaba.mnnllm.android.chat.model

import android.content.Context
import android.database.sqlite.SQLiteDatabase
import androidx.test.core.app.ApplicationProvider
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class ChatDatabaseMigrationTest {
    private val context: Context get() = ApplicationProvider.getApplicationContext()

    @Before
    fun setUp() {
        context.deleteDatabase("chat.db")
    }

    @After
    fun tearDown() {
        context.deleteDatabase("chat.db")
    }

    private fun columns(db: SQLiteDatabase, table: String): Set<String> =
        db.rawQuery("PRAGMA table_info($table)", null).use { cursor ->
            buildSet { while (cursor.moveToNext()) add(cursor.getString(1)) }
        }

    @Test
    fun `v8 database upgrades in place and keeps its rows`() {
        val path = context.getDatabasePath("chat.db").apply { parentFile?.mkdirs() }
        SQLiteDatabase.openOrCreateDatabase(path, null).use { db ->
            db.execSQL("CREATE TABLE Session (sessionId TEXT PRIMARY KEY, modelId TEXT, name TEXT, lastChatTime INTEGER DEFAULT 0)")
            db.execSQL("CREATE TABLE ChatData (_id INTEGER PRIMARY KEY AUTOINCREMENT, sessionId TEXT, time TEXT, type INTEGER, " +
                "text TEXT, imageUri TEXT, audioUri TEXT, audioDuration REAL, videoUri TEXT, displayText TEXT, reserve1 TEXT, " +
                "reserve2 TEXT, thinkingText TEXT, thinkingFinishedTime INTEGER)")
            db.execSQL("CREATE TABLE DownloadHistory (modelId TEXT PRIMARY KEY, downloadTime INTEGER, modelPath TEXT, modelType TEXT DEFAULT 'LLM')")
            db.execSQL("INSERT INTO Session (sessionId, modelId, name) VALUES ('s1', 'm1', 'old')")
            db.execSQL("INSERT INTO ChatData (sessionId, type, text) VALUES ('s1', 2, 'hello')")
            db.version = 8
        }

        ChatDatabaseHelper(context).writableDatabase.use { db ->
            assertEquals(9, db.version)
            assertTrue(columns(db, "ChatData").contains("benchmarkInfo"))
            assertTrue(columns(db, "Session").containsAll(listOf("contextSummary", "contextSummaryAnchorId")))
            db.rawQuery("SELECT text FROM ChatData WHERE sessionId='s1'", null).use {
                assertTrue(it.moveToFirst())
                assertEquals("hello", it.getString(0))
            }
        }
    }

    @Test
    fun `fresh database has the new columns`() {
        ChatDatabaseHelper(context).writableDatabase.use { db ->
            assertTrue(columns(db, "ChatData").contains("benchmarkInfo"))
            assertTrue(columns(db, "Session").contains("contextSummary"))
        }
    }
}
