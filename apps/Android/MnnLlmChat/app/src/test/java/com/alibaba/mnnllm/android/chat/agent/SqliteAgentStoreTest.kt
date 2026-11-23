package com.alibaba.mnnllm.android.chat.agent

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.util.UUID

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [26], application = android.app.Application::class)
@OptIn(ExperimentalCoroutinesApi::class)
class SqliteAgentStoreTest {
    @Test
    fun `SQLite survives reopening shares memories across sessions and supports separate deletion`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val consent = AgentConsentController()
        val revision = consent.confirmEnable().revision
        val name = "agent-test-${UUID.randomUUID()}.sqlite3"
        val dispatcher = StandardTestDispatcher(testScheduler)
        var store = SqliteAgentStore(context, consent, name, dispatcher, now = { 1234L })
        assertFalse(store.isAgentSession("first"))
        assertTrue(store.setAgentSession("first", true, revision))
        assertFalse(store.isAgentSession("second"))
        assertTrue(store.saveMemory("我喜欢茶", "preference", revision))
        assertTrue(store.saveSkill(AgentSkill("发布", listOf("发布"), "先测试再发布"), revision))
        store.close()
        store = SqliteAgentStore(context, consent, name, dispatcher)
        assertTrue(store.isAgentSession("first"))
        assertEquals(AgentMemory("我喜欢茶", "preference", 1234), store.allMemories().single())
        assertEquals("先测试再发布", store.allSkills().single().actionTemplate)
        store.deleteMemory("我喜欢茶")
        assertTrue(store.allMemories().isEmpty())
        assertEquals(1, store.allSkills().size)
        store.deleteSkill("发布")
        assertTrue(store.allSkills().isEmpty())
        store.setAgentSession("first", false, revision)
        assertFalse(store.isAgentSession("first"))
        store.close()
    }

    @Test
    fun `limits replace duplicates and evict oldest memory and skill rows atomically`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val consent = AgentConsentController()
        val revision = consent.confirmEnable().revision
        var time = 0L
        val store = SqliteAgentStore(context, consent, "agent-test-${UUID.randomUUID()}.sqlite3",
            StandardTestDispatcher(testScheduler), now = { ++time })
        for (index in 0..34) {
            store.saveMemory("memory $index", "fact", revision)
            store.saveSkill(AgentSkill("skill $index", listOf("trigger"), "action $index"), revision)
        }
        assertEquals(30, store.allMemories().size)
        assertEquals(30, store.allSkills().size)
        assertFalse(store.allMemories().any { it.content == "memory 0" })
        assertFalse(store.allSkills().any { it.name == "skill 0" })
        store.saveMemory("memory 20", "preference", revision)
        assertEquals("memory 20", store.allMemories().first().content)
        assertEquals("preference", store.allMemories().first().category)
        assertEquals(30, store.allMemories().size)
        store.saveMemory("x".repeat(500), "", revision)
        store.saveSkill(AgentSkill("y".repeat(100), (1..20).map { "trigger $it" }, "z".repeat(600)), revision)
        assertEquals(168, store.allMemories().first().content.length)
        assertEquals(64, store.allSkills().first().name.length)
        assertEquals(350, store.allSkills().first().actionTemplate.length)
        assertEquals(8, store.allSkills().first().triggers.size)
        store.clearMemories()
        assertTrue(store.allMemories().isEmpty())
        assertFalse(store.allSkills().isEmpty())
        store.clearSkills()
        assertTrue(store.allSkills().isEmpty())
        store.close()
    }

    @Test
    fun `no memory or skill write is possible without current persistence consent`() = runTest {
        val consent = AgentConsentController()
        val store = SqliteAgentStore(ApplicationProvider.getApplicationContext(), consent,
            "agent-test-${UUID.randomUUID()}.sqlite3", StandardTestDispatcher(testScheduler))
        val skill = AgentSkill("name", listOf("trigger"), "action")
        assertFalse(store.saveMemory("not saved", "fact", 0))
        assertFalse(store.saveSkill(skill, 0))
        assertFalse(store.setAgentSession("first", true, 0))
        val noPersistence = consent.confirmEnable(persistenceAllowed = false).revision
        assertFalse(store.saveMemory("not saved", "fact", noPersistence))
        assertFalse(store.saveSkill(skill, noPersistence))
        val revision = consent.confirmEnable().revision
        assertTrue(store.saveMemory("saved", "fact", revision))
        assertTrue(store.saveSkill(skill, revision))
        consent.disable()
        assertFalse(store.saveMemory("not saved", "fact", revision))
        assertFalse(store.saveSkill(skill, revision))
        consent.confirmEnable()
        assertFalse(store.saveMemory("stale consent", "fact", revision))
        assertEquals(listOf("saved"), store.allMemories().map { it.content })
        store.clearAll()
        assertTrue(store.allMemories().isEmpty())
        assertTrue(store.allSkills().isEmpty())
        store.close()
    }

    @Test
    fun `SQL parameters preserve punctuation and do not execute content`() = runTest {
        val consent = AgentConsentController()
        val revision = consent.confirmEnable().revision
        val store = SqliteAgentStore(ApplicationProvider.getApplicationContext(), consent,
            "agent-test-${UUID.randomUUID()}.sqlite3", StandardTestDispatcher(testScheduler))
        val content = "'); DROP TABLE agentSkills; --"
        assertTrue(store.saveMemory(content, "fact", revision))
        assertTrue(store.saveSkill(AgentSkill("name'", listOf("a\nb", "a\nb"), "action"), revision))
        assertEquals(content, store.allMemories().single().content)
        assertEquals(listOf("a b"), store.allSkills().single().triggers)
        store.deleteMemory(content)
        assertTrue(store.allMemories().isEmpty())
        assertEquals(1, store.allSkills().size)
        store.close()
    }
}
