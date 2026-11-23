package com.alibaba.mnnllm.android.chat.agent

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.time.Instant

class SqliteAgentStore(
    context: Context,
    private val consent: AgentConsentController,
    databaseName: String = "agent.sqlite3",
    private val dispatcher: CoroutineDispatcher = Dispatchers.IO,
    private val now: () -> Long = System::currentTimeMillis,
) : AgentStore {
    private val helper = Helper(context.applicationContext, databaseName)
    private val mutex = Mutex()

    private suspend fun <T> read(block: (SQLiteDatabase) -> T): T = withContext(dispatcher) {
        mutex.withLock {
            currentCoroutineContext().ensureActive()
            block(helper.readableDatabase)
        }
    }

    private suspend fun write(consentRevision: Long? = null, persistence: Boolean = true,
        block: (SQLiteDatabase) -> Unit): Boolean = withContext(dispatcher) {
        mutex.withLock {
            currentCoroutineContext().ensureActive()
            if (consentRevision != null && !consent.isCurrent(consentRevision)) return@withLock false
            if (consentRevision != null && persistence && !consent.canPersist(consentRevision)) return@withLock false
            val db = helper.writableDatabase
            val job = currentCoroutineContext()
            val operation = {
                job.ensureActive()
                db.beginTransaction()
                try {
                    block(db)
                    job.ensureActive()
                    db.setTransactionSuccessful()
                } finally { db.endTransaction() }
                true
            }
            if (consentRevision == null) operation()
            else consent.withPermit(consentRevision, persistence, operation) ?: false
        }
    }

    override suspend fun isAgentSession(sessionId: String): Boolean = read { db ->
        db.rawQuery("SELECT 1 FROM agentSessions WHERE historyId=?", arrayOf(sessionId)).use { it.moveToFirst() }
    }

    override suspend fun setAgentSession(sessionId: String, enabled: Boolean, consentRevision: Long): Boolean {
        require(sessionId.isNotBlank() && sessionId.length <= 256)
        return write(if (enabled) consentRevision else null, persistence = false) { db ->
            if (enabled) db.insertWithOnConflict("agentSessions", null, ContentValues().apply {
                put("historyId", sessionId)
            }, SQLiteDatabase.CONFLICT_IGNORE).also { check(it != -1L || isSessionPresent(db, sessionId)) }
            else db.delete("agentSessions", "historyId=?", arrayOf(sessionId))
        }
    }

    private fun isSessionPresent(db: SQLiteDatabase, id: String): Boolean =
        db.rawQuery("SELECT 1 FROM agentSessions WHERE historyId=?", arrayOf(id)).use { it.moveToFirst() }

    override suspend fun allMemories(): List<AgentMemory> = read { db ->
        db.rawQuery("SELECT content,category,updatedAt FROM agentMemories ORDER BY updatedAt DESC,rowid DESC LIMIT ?",
            arrayOf(AgentStoreLimits.MAX_MEMORIES.toString())).use { cursor ->
            buildList {
                while (cursor.moveToNext()) add(AgentMemory(cursor.getString(0), cursor.getString(1), cursor.getLong(2)))
            }
        }
    }

    override suspend fun allSkills(): List<AgentSkill> = read { db ->
        db.rawQuery("SELECT name,triggers,actionTemplate,updatedAt FROM agentSkills ORDER BY updatedAt DESC,rowid DESC LIMIT ?",
            arrayOf(AgentStoreLimits.MAX_SKILLS.toString())).use { cursor ->
            buildList {
                while (cursor.moveToNext()) add(AgentSkill(cursor.getString(0),
                    cursor.getString(1).split('\n').filter { it.isNotEmpty() }, cursor.getString(2), cursor.getLong(3)))
            }
        }
    }

    override suspend fun saveMemory(content: String, category: String, consentRevision: Long): Boolean {
        val text = AgentStoreLimits.memory(content)
        return write(consentRevision) { db ->
            val values = ContentValues().apply {
                put("content", text)
                put("category", category.trim().take(32).ifEmpty { "general" })
                put("updatedAt", now())
            }
            check(db.insertWithOnConflict("agentMemories", null, values, SQLiteDatabase.CONFLICT_REPLACE) != -1L)
            db.execSQL("DELETE FROM agentMemories WHERE rowid NOT IN (SELECT rowid FROM agentMemories ORDER BY updatedAt DESC,rowid DESC LIMIT ?)",
                arrayOf(AgentStoreLimits.MAX_MEMORIES))
        }
    }

    override suspend fun saveSkill(skill: AgentSkill, consentRevision: Long): Boolean {
        val bounded = AgentStoreLimits.skill(skill)
        return write(consentRevision) { db ->
            val values = ContentValues().apply {
                put("name", bounded.name)
                put("triggers", bounded.triggers.joinToString("\n"))
                put("actionTemplate", bounded.actionTemplate)
                put("updatedAt", now())
            }
            check(db.insertWithOnConflict("agentSkills", null, values, SQLiteDatabase.CONFLICT_REPLACE) != -1L)
            db.execSQL("DELETE FROM agentSkills WHERE rowid NOT IN (SELECT rowid FROM agentSkills ORDER BY updatedAt DESC,rowid DESC LIMIT ?)",
                arrayOf(AgentStoreLimits.MAX_SKILLS))
        }
    }

    override suspend fun deleteMemory(content: String) { write { it.delete("agentMemories", "content=?", arrayOf(content)) } }
    override suspend fun deleteSkill(name: String) { write { it.delete("agentSkills", "name=?", arrayOf(name)) } }
    override suspend fun clearMemories() { write { it.delete("agentMemories", null, null) } }
    override suspend fun clearSkills() { write { it.delete("agentSkills", null, null) } }
    override suspend fun clearAll() {
        write { db ->
            db.delete("agentMemories", null, null)
            db.delete("agentSkills", null, null)
        }
    }

    override suspend fun saveToolSteps(sessionId: String, record: AgentToolStepRecord, consentRevision: Long): Boolean {
        require(sessionId.isNotBlank() && sessionId.length <= 256)
        require(record.assistantId >= -1 && record.time.length <= 64)
        Instant.parse(record.time)
        require(record.phase in setOf(AgentPhase.COMPLETED, AgentPhase.CANCELLED, AgentPhase.FAILED))
        val steps = record.steps.take(64).map { step ->
            val canonical = AgentStoredStep.from(AgentStep(0, step.kind, "", status = step.status))
            canonical.copy(status = if (step.status != AgentStepStatus.RUNNING) step.status else
                if (record.phase == AgentPhase.CANCELLED) AgentStepStatus.CANCELLED else AgentStepStatus.FAILED)
        }
        return write(consentRevision, persistence = false) { db ->
            db.delete("agentToolSteps", "historyId=? AND turnTime=?", arrayOf(sessionId, record.time))
            steps.forEachIndexed { index, step ->
                val values = ContentValues().apply {
                    put("historyId", sessionId)
                    put("assistantId", record.assistantId)
                    put("turnTime", record.time)
                    put("stepIndex", index)
                    put("phase", record.phase.name)
                    put("kind", step.kind.name)
                    put("tool", step.tool)
                    put("title", step.title)
                    put("status", step.status.name)
                }
                check(db.insertOrThrow("agentToolSteps", null, values) != -1L)
            }
        }
    }

    override suspend fun toolStepsForSession(sessionId: String): List<AgentToolStepRecord> = read { db ->
        db.rawQuery("SELECT assistantId,turnTime,phase,kind,tool,title,status FROM agentToolSteps WHERE historyId=? ORDER BY turnTime,stepIndex",
            arrayOf(sessionId)).use { cursor ->
            val turns = linkedMapOf<String, AgentToolStepRecord>()
            while (cursor.moveToNext()) {
                val phase = AgentPhase.entries.firstOrNull { it.name == cursor.getString(2) } ?: continue
                val kind = AgentStepKind.entries.firstOrNull { it.name == cursor.getString(3) } ?: continue
                val status = AgentStepStatus.entries.firstOrNull { it.name == cursor.getString(6) } ?: continue
                val step = AgentStoredStep.from(AgentStep(0, kind, "", status = status))
                if (step.tool != cursor.getString(4) || step.title != cursor.getString(5)) continue
                val time = cursor.getString(1)
                val turn = turns[time] ?: AgentToolStepRecord(cursor.getLong(0), time, phase, emptyList())
                turns[time] = turn.copy(steps = turn.steps + step)
            }
            turns.values.toList()
        }
    }

    override suspend fun clearToolSteps(sessionId: String) {
        write { it.delete("agentToolSteps", "historyId=?", arrayOf(sessionId)) }
    }

    suspend fun close() = withContext(dispatcher) { mutex.withLock { helper.close() } }

    private class Helper(context: Context, name: String) : SQLiteOpenHelper(context, name, null, 2) {
        override fun onCreate(db: SQLiteDatabase) {
            db.execSQL("CREATE TABLE agentSessions(historyId TEXT PRIMARY KEY NOT NULL)")
            db.execSQL("CREATE TABLE agentMemories(content TEXT PRIMARY KEY NOT NULL,category TEXT NOT NULL,updatedAt INTEGER NOT NULL)")
            db.execSQL("CREATE TABLE agentSkills(name TEXT PRIMARY KEY NOT NULL,triggers TEXT NOT NULL,actionTemplate TEXT NOT NULL,updatedAt INTEGER NOT NULL)")
            createToolSteps(db)
        }
        private fun createToolSteps(db: SQLiteDatabase) {
            db.execSQL("CREATE TABLE IF NOT EXISTS agentToolSteps(historyId TEXT NOT NULL,assistantId INTEGER NOT NULL,turnTime TEXT NOT NULL,stepIndex INTEGER NOT NULL,phase TEXT NOT NULL,kind TEXT NOT NULL,tool TEXT NOT NULL,title TEXT NOT NULL,status TEXT NOT NULL,PRIMARY KEY(historyId,turnTime,stepIndex))")
        }

        override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
            check(oldVersion == 1 && newVersion == 2) { "Unsupported agent database upgrade $oldVersion to $newVersion" }
            createToolSteps(db)
        }
    }
}
