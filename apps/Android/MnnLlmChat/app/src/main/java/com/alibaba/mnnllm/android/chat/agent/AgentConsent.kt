package com.alibaba.mnnllm.android.chat.agent

import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.util.Locale

data class AgentConsentState(
    val enabled: Boolean = false,
    val networkAllowed: Boolean = false,
    val persistenceAllowed: Boolean = false,
    val revision: Long = 0,
)

class AgentConsentController(private val dispatcher: CoroutineDispatcher = Dispatchers.IO) {
    private val lock = Any()
    private val mutableState = MutableStateFlow(AgentConsentState())
    val state: StateFlow<AgentConsentState> = mutableState.asStateFlow()

    suspend fun confirmEnable(networkAllowed: Boolean = true, persistenceAllowed: Boolean = true): AgentConsentState =
        withContext(dispatcher) {
            val context = currentCoroutineContext()
            synchronized(lock) {
                context.ensureActive()
                AgentConsentState(true, networkAllowed, persistenceAllowed, mutableState.value.revision + 1)
                    .also { mutableState.value = it }
            }
        }

    suspend fun disable() = withContext(dispatcher) {
        val context = currentCoroutineContext()
        synchronized(lock) {
            context.ensureActive()
            mutableState.value = AgentConsentState(revision = mutableState.value.revision + 1)
        }
    }

    fun isCurrent(revision: Long): Boolean = state.value.let { it.enabled && it.revision == revision }
    fun canUseNetwork(revision: Long): Boolean = state.value.let { it.enabled && it.revision == revision && it.networkAllowed }
    fun canPersist(revision: Long): Boolean = state.value.let { it.enabled && it.revision == revision && it.persistenceAllowed }

    internal fun <T> withPermit(revision: Long, persistence: Boolean, block: () -> T): T? = synchronized(lock) {
        if (!isCurrent(revision) || (persistence && !state.value.persistenceAllowed)) null else block()
    }
}

object AgentInputPolicy {
    private val memoryPrefixes = listOf("请记住", "记住", "记一下", "请记得", "记得", "remember that", "remember")
    private val procedureMarkers = listOf("保存技能", "记住步骤", "记住流程", "保存流程", "save skill", "remember these steps", "save procedure")
    private val storageDenials = listOf("不要记住", "不要保存", "不要记忆", "别记住", "别保存", "不需要记住",
        "don't remember", "do not remember", "don't save", "do not save")
    private val fillers = listOf("请帮我", "帮我", "帮忙", "麻烦", "请你", "请", "联网", "上网", "网上", "搜索一下", "搜一下",
        "搜索", "查一下", "查一查", "查查", "查询", "看一下", "看下", "告诉我", "一下", "能不能", "可以")
    private val englishFillers = setOf("what", "is", "it", "the", "a", "an", "of", "for", "to", "in", "and", "does",
        "do", "how", "about", "are", "which", "who", "me", "please", "search", "find", "look", "up")

    fun explicitMemory(request: String): String? {
        val trimmed = request.trim()
        val prefix = memoryPrefixes.firstOrNull { trimmed.startsWith(it, ignoreCase = true) } ?: return null
        return trimmed.drop(prefix.length).trim(' ', '\n', '\t', '：', ':', '，', ',', '。', '.')
            .takeIf { it.length >= 2 }?.take(AgentStoreLimits.MAX_MEMORY_LENGTH)
    }

    private fun normalized(text: String): String = text.trim().lowercase(Locale.ROOT).replace(Regex("\\s+"), " ")

    fun memoryIsGrounded(content: String, request: String): Boolean {
        val text = normalized(content)
        return text.length >= 2 && storageDenials.none { request.contains(it, ignoreCase = true) } && normalized(request).contains(text)
    }

    fun skillIsGrounded(skill: AgentSkill, request: String): Boolean =
        procedureMarkers.any { request.contains(it, ignoreCase = true) } &&
            skill.name.isNotBlank() && memoryIsGrounded(skill.actionTemplate, request) &&
            skill.triggers.isNotEmpty() && skill.triggers.all {
                val keyword = normalized(it)
                keyword.isNotEmpty() && normalized(request).contains(keyword)
            }

    fun coreSearchQuery(request: String): String {
        var text = request.take(2048)
        for (filler in fillers) text = text.replace(filler, " ")
        text = text.replace(Regex("https?://\\S+", RegexOption.IGNORE_CASE), " ")
        val words = text.replace(Regex("[^\\p{L}\\p{N}\\s_.-]"), " ")
            .split(Regex("\\s+")).filter { it.isNotBlank() && it.lowercase(Locale.ROOT) !in englishFillers }
        return words.joinToString(" ").take(80).trim()
    }

    fun privateSearchQuery(modelQuery: String, request: String): String {
        val query = modelQuery.trim()
        val source = normalized(request)
        val tokens = query.split(Regex("[^\\p{L}\\p{N}_.-]+")).filter { it.isNotBlank() }
        val sourceTokens = source.split(Regex("[^\\p{L}\\p{N}_.-]+"))
        val grounded = tokens.isNotEmpty() && tokens.all { token ->
            val word = token.lowercase(Locale.ROOT)
            if (word.any { it.code in 0x3400..0x9fff }) word.length >= 2 && source.contains(word)
            else word in sourceTokens
        }
        return if (grounded && query.length <= 80 && !query.contains(Regex("https?://|@", RegexOption.IGNORE_CASE))) query
            else coreSearchQuery(request)
    }

    fun resultNumber(value: String): Int? = value.trim(' ', '[', ']', '(', ')', '#', '结', '果', '第', '条', '.', '。')
        .takeIf { it.isNotEmpty() && it.all { ch -> ch in '0'..'9' } }?.toIntOrNull()?.takeIf { it > 0 }
}
