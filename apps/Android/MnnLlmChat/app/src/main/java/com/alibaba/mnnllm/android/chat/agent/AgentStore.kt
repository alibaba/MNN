package com.alibaba.mnnllm.android.chat.agent

data class AgentMemory(val content: String, val category: String, val updatedAt: Long)
data class AgentSkill(
    val name: String,
    val triggers: List<String>,
    val actionTemplate: String,
    val updatedAt: Long = 0,
)

data class AgentStoredStep(
    val kind: AgentStepKind,
    val tool: String,
    val title: String,
    val status: AgentStepStatus,
) {
    companion object {
        fun from(step: AgentStep): AgentStoredStep {
            val tool = when (step.kind) {
                AgentStepKind.PREPARING -> "preparing_tool"
                AgentStepKind.TIME -> "get_current_time"
                AgentStepKind.CALCULATOR -> "calculator"
                AgentStepKind.SEARCH -> "web_search"
                AgentStepKind.READ_PAGE -> "read_webpage"
                AgentStepKind.MEMORY -> "save_memory"
                AgentStepKind.SKILL -> "save_skill"
                AgentStepKind.OTHER -> "unknown_tool"
            }
            return AgentStoredStep(step.kind, tool, tool, step.status)
        }
    }
}

data class AgentToolStepRecord(
    val assistantId: Long,
    val time: String,
    val phase: AgentPhase,
    val steps: List<AgentStoredStep>,
)

object AgentStoreLimits {
    const val MAX_MEMORIES = 30
    const val MAX_SKILLS = 30
    const val MAX_MEMORY_LENGTH = 168
    const val MAX_SKILL_LENGTH = 350
    const val MAX_NAME_LENGTH = 64
    const val MAX_TRIGGERS = 8

    fun memory(content: String): String = content.trim().take(MAX_MEMORY_LENGTH).also {
        require(it.isNotEmpty() && it.none { ch -> ch.code < 32 && ch != '\n' && ch != '\t' }) { "Invalid memory" }
    }

    fun skill(skill: AgentSkill): AgentSkill {
        val name = skill.name.trim().take(MAX_NAME_LENGTH)
        val action = skill.actionTemplate.trim().take(MAX_SKILL_LENGTH)
        require(name.isNotEmpty() && action.isNotEmpty()) { "Invalid skill" }
        require((name + action).none { it.code < 32 && it != '\n' && it != '\t' }) { "Invalid skill" }
        return skill.copy(name = name, actionTemplate = action, triggers = skill.triggers.map {
            it.trim().replace(Regex("[\\r\\n\\t]"), " ").take(MAX_NAME_LENGTH)
        }.filter { it.isNotEmpty() }.distinct().take(MAX_TRIGGERS))
    }
}

interface AgentStore {
    suspend fun isAgentSession(sessionId: String): Boolean
    suspend fun setAgentSession(sessionId: String, enabled: Boolean, consentRevision: Long): Boolean
    suspend fun allMemories(): List<AgentMemory>
    suspend fun allSkills(): List<AgentSkill>
    suspend fun saveMemory(content: String, category: String, consentRevision: Long): Boolean
    suspend fun saveSkill(skill: AgentSkill, consentRevision: Long): Boolean
    suspend fun deleteMemory(content: String)
    suspend fun deleteSkill(name: String)
    suspend fun clearMemories()
    suspend fun clearSkills()
    suspend fun clearAll()
    suspend fun saveToolSteps(sessionId: String, record: AgentToolStepRecord, consentRevision: Long): Boolean = false
    suspend fun toolStepsForSession(sessionId: String): List<AgentToolStepRecord> = emptyList()
    suspend fun clearToolSteps(sessionId: String) { }
}
