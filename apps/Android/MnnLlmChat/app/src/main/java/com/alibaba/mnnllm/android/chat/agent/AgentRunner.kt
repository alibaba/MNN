package com.alibaba.mnnllm.android.chat.agent

import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.withTimeoutOrNull
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.withContext
import java.nio.charset.StandardCharsets
import java.time.ZonedDateTime
import java.util.Locale

enum class AgentPhase { IDLE, RUNNING, COMPLETED, CANCELLED, FAILED }
enum class AgentStepStatus { RUNNING, DONE, FAILED, SKIPPED, CANCELLED }
enum class AgentStepKind { PREPARING, TIME, CALCULATOR, SEARCH, READ_PAGE, MEMORY, SKILL, OTHER }

data class AgentStep(
    val id: Int,
    val kind: AgentStepKind,
    val title: String,
    val detail: String = "",
    val status: AgentStepStatus = AgentStepStatus.RUNNING,
    val result: String = "",
)

data class AgentRunState(
    val phase: AgentPhase = AgentPhase.IDLE,
    val pass: Int = 0,
    val answer: String = "",
    val thinking: String = "",
    val steps: List<AgentStep> = emptyList(),
    val toolCalls: Int = 0,
    val pageReads: Int = 0,
    val droppedHistoryMessages: Int = 0,
    val droppedToolRounds: Int = 0,
    val toolResultsTruncated: Boolean = false,
    val contextTokens: Int = 0,
    val contextTokensEstimated: Boolean = false,
    val error: String? = null,
    val metrics: Map<String, Any> = emptyMap(),
)

data class AgentRequest(
    val userText: String,
    val history: List<AgentMessage> = emptyList(),
    val baseSystemPrompt: String = "",
    val contextLimitTokens: Int = 8192,
    val outputReserveTokens: Int = 1024,
    val expectsThinking: Boolean = false,
    val sessionId: String? = null,
)

class AgentContextBudgetException : IllegalArgumentException("Input exceeds the agent context budget")

class AgentRunner(
    private val inference: AgentInference,
    private val tools: AgentToolExecutor,
    private val store: AgentStore,
    private val consent: AgentConsentController,
    private val budget: AgentBudget = AgentBudget(),
    private val dispatcher: CoroutineDispatcher = Dispatchers.Default,
    private val now: () -> ZonedDateTime = { ZonedDateTime.now() },
) {
    private val mutableState = MutableStateFlow(AgentRunState())
    val state: StateFlow<AgentRunState> = mutableState.asStateFlow()
    private val runMutex = Mutex()
    private val urlPolicy = AgentUrlPolicy()

    private data class Exchange(val assistant: String, val observations: List<String>, val remainingCalls: Int,
        var characterLimit: Int = 4000) {
        fun messages(finalOnly: Boolean = false): List<AgentMessage> =
            listOf(AgentMessage(AgentRole.ASSISTANT, assistant.take(4096)),
                AgentMessage(AgentRole.USER, AgentPrompts.toolResultsMessage(observations.map {
                    if (it.length > characterLimit) it.take(characterLimit) + "\n[TRUNCATED_TOOL_RESULT]" else it
                }, if (finalOnly) 0 else remainingCalls)))
    }

    suspend fun run(request: AgentRequest, onState: (AgentRunState) -> Unit = {}): AgentRunState {
        val grant = consent.state.value
        require(request.userText.isNotBlank()) { "Empty request" }
        require(request.contextLimitTokens in 256..1048576 && request.outputReserveTokens >= 0)
        require(request.history.all { it.role != AgentRole.SYSTEM }) { "History must not contain system messages" }
        require(request.sessionId == null || request.sessionId.isNotBlank() && request.sessionId.length <= 256)
        check(grant.enabled) { "Agent mode requires user confirmation" }
        check(runMutex.tryLock()) { "An agent request is already running" }
        try {
            return withContext(dispatcher) {
                coroutineScope {
                    val runJob = currentCoroutineContext()
                    val watcher = launch {
                        consent.state.first { it.revision != grant.revision || !it.enabled }
                        runJob[Job]?.cancel(CancellationException("Agent consent changed"))
                    }
                    try { execute(request, grant, onState) }
                    finally { watcher.cancel() }
                }
            }
        } catch (e: CancellationException) {
            val cancelled = mutableState.value.copy(phase = AgentPhase.CANCELLED, steps = mutableState.value.steps.map {
                if (it.status == AgentStepStatus.RUNNING) it.copy(status = AgentStepStatus.CANCELLED, result = "Cancelled") else it
            })
            mutableState.value = cancelled
            onState(cancelled)
            throw e
        } catch (e: Exception) {
            val message = if (e is AgentContextBudgetException) e.message
                else "Agent request failed (${e.javaClass.simpleName}: ${e.message})"
            val failed = mutableState.value.copy(phase = AgentPhase.FAILED, error = message,
                metrics = if (e is AgentInferenceException) e.metrics else mutableState.value.metrics,
                steps = mutableState.value.steps.map {
                if (it.status == AgentStepStatus.RUNNING) it.copy(status = AgentStepStatus.FAILED, result = "Failed") else it
            })
            mutableState.value = failed
            onState(failed)
            return failed
        } finally { runMutex.unlock() }
    }

    private suspend fun execute(request: AgentRequest, grant: AgentConsentState, onState: (AgentRunState) -> Unit): AgentRunState {
        var state = AgentRunState(phase = AgentPhase.RUNNING)
        val steps = mutableListOf<AgentStep>()
        var nextStepId = 0
        val thoughts = mutableListOf<String>()
        fun publish() {
            state = state.copy(steps = steps.toList())
            mutableState.value = state
            onState(state)
        }
        fun addStep(kind: AgentStepKind, title: String, detail: String, status: AgentStepStatus = AgentStepStatus.RUNNING,
            result: String = ""): Int {
            val id = nextStepId++
            steps.add(AgentStep(id, kind, title, detail.take(2048), status, result.take(240)))
            publish()
            return id
        }
        fun finishStep(id: Int, status: AgentStepStatus, result: String) {
            val index = steps.indexOfFirst { it.id == id }
            if (index >= 0) steps[index] = steps[index].copy(status = status, result = result.take(240))
            publish()
        }
        suspend fun checkActive() {
            currentCoroutineContext().ensureActive()
            if (!consent.isCurrent(grant.revision)) throw CancellationException("Agent consent changed")
        }
        publish()
        checkActive()
        request.sessionId?.let { check(store.isAgentSession(it)) { "Agent mode is disabled for this session" } }
        val memories = store.allMemories()
        val skills = AgentPrompts.matchingSkills(store.allSkills(), request.userText)
        val date = now()
        var promptMemories = memories.take(8)
        var promptSkills = skills
        val history = request.history.toMutableList()
        val exchanges = mutableListOf<Exchange>()
        val allowedUrls = (urlPolicy.urlsInUserText(request.userText) + request.history.filter { it.role == AgentRole.USER }
            .flatMap { urlPolicy.urlsInUserText(it.content) }).toMutableSet()
        var resultUrls = emptyList<String>()
        val seen = mutableSetOf<String>()
        var searched = false
        var finalOnly = false
        var thinkingAllowed = request.expectsThinking
        var retriedThinking = false
        var memorySaved = false
        var toolRounds = 0

        suspend fun persistMemory(content: String, category: String): String {
            checkActive()
            val id = addStep(AgentStepKind.MEMORY, "save_memory", content)
            if (!consent.canPersist(grant.revision) || !AgentInputPolicy.memoryIsGrounded(content, request.userText)) {
                finishStep(id, AgentStepStatus.SKIPPED, "Consent or user-source evidence is missing")
                return "[TOOL_ERROR] Memory was not saved: consent or current-user evidence is missing"
            }
            val bounded = try { AgentStoreLimits.memory(content) } catch (_: IllegalArgumentException) {
                finishStep(id, AgentStepStatus.FAILED, "Invalid memory")
                return "[TOOL_ERROR] Invalid memory"
            }
            if ("memory:$bounded" in seen) {
                finishStep(id, AgentStepStatus.SKIPPED, "Duplicate call")
                return "[TOOL_ERROR] Duplicate memory call was skipped"
            }
            val result = try {
                if (store.saveMemory(bounded, category, grant.revision)) AgentToolResult("[MEMORY_SAVED] $bounded")
                else AgentToolResult.error("Memory consent was revoked")
            } catch (e: CancellationException) { throw e }
            catch (_: Exception) { AgentToolResult.error("Memory storage failed") }
            checkActive()
            if (result.success) {
                memorySaved = true
                seen.add("memory:$bounded")
            }
            finishStep(id, if (result.success) AgentStepStatus.DONE else AgentStepStatus.FAILED,
                if (result.success) "Saved locally" else result.observation)
            return result.observation
        }

        suspend fun persistSkill(skill: AgentSkill): String {
            checkActive()
            val id = addStep(AgentStepKind.SKILL, "save_skill", skill.name)
            if (!consent.canPersist(grant.revision) || !AgentInputPolicy.skillIsGrounded(skill, request.userText)) {
                finishStep(id, AgentStepStatus.SKIPPED, "Consent or explicit user procedure is missing")
                return "[TOOL_ERROR] Skill was not saved: no permitted user-authored procedure"
            }
            val bounded = try { AgentStoreLimits.skill(skill) } catch (_: IllegalArgumentException) {
                finishStep(id, AgentStepStatus.FAILED, "Invalid skill")
                return "[TOOL_ERROR] Invalid skill"
            }
            if (!seen.add("skill:${bounded.name}:${bounded.actionTemplate}")) {
                finishStep(id, AgentStepStatus.SKIPPED, "Duplicate call")
                return "[TOOL_ERROR] Duplicate skill update was skipped"
            }
            val result = try {
                if (store.saveSkill(bounded, grant.revision)) AgentToolResult("[SKILL_SAVED] ${bounded.name}")
                else AgentToolResult.error("Skill consent was revoked")
            } catch (e: CancellationException) { throw e }
            catch (_: Exception) { AgentToolResult.error("Skill storage failed") }
            checkActive()
            finishStep(id, if (result.success) AgentStepStatus.DONE else AgentStepStatus.FAILED,
                if (result.success) "Saved locally" else result.observation)
            return result.observation
        }

        var pass = 0
        while (pass < budget.maxPasses + 2) {
            checkActive()
            pass++
            state = state.copy(pass = pass, answer = "")
            publish()
            if (finalOnly) thinkingAllowed = false
            fun messages(): List<AgentMessage> {
                val system = if (finalOnly) AgentPrompts.answerSystemPrompt(promptMemories, date, request.baseSystemPrompt)
                    else AgentPrompts.systemPrompt(promptMemories, promptSkills, grant, date, request.baseSystemPrompt)
                var modelRequest = AgentPrompts.requestWithMemories(request.userText, promptMemories)
                if (finalOnly && exchanges.isEmpty()) modelRequest += "\n" + AgentPrompts.finalAnswerMessage
                return listOf(AgentMessage(AgentRole.SYSTEM, system)) + history +
                    AgentMessage(AgentRole.USER, modelRequest) + exchanges.flatMapIndexed { index, exchange ->
                        exchange.messages(finalOnly && index == exchanges.lastIndex)
                    }
            }
            val outputLimit = minOf(maxOf(1, request.outputReserveTokens), request.contextLimitTokens / 2)
            val inputLimit = request.contextLimitTokens - outputLimit
            var prompts = messages()
            var tokenCount = countTokens(prompts)
            while (tokenCount.first > inputLimit) {
                checkActive()
                when {
                    history.isNotEmpty() -> {
                        var count = 1
                        while (count < history.size && history[count].role != AgentRole.USER) count++
                        history.subList(0, count).clear()
                        state = state.copy(droppedHistoryMessages = state.droppedHistoryMessages + count)
                    }
                    exchanges.size > 1 -> {
                        exchanges.removeAt(0)
                        state = state.copy(droppedToolRounds = state.droppedToolRounds + 1)
                    }
                    promptSkills.isNotEmpty() -> promptSkills = promptSkills.dropLast(1)
                    promptMemories.isNotEmpty() -> promptMemories = promptMemories.dropLast(1)
                    exchanges.isNotEmpty() && !finalOnly -> finalOnly = true
                    exchanges.lastOrNull()?.let { it.characterLimit > 256 } == true -> {
                        exchanges.last().characterLimit = maxOf(256, exchanges.last().characterLimit / 2)
                        state = state.copy(toolResultsTruncated = true)
                    }
                    else -> throw AgentContextBudgetException()
                }
                prompts = messages()
                tokenCount = countTokens(prompts)
            }
            state = state.copy(contextTokens = tokenCount.first, contextTokensEstimated = tokenCount.second)
            if (finalOnly) thinkingAllowed = false
            publish()
            val buffer = StringBuilder()
            var pendingId: Int? = null
            val context = currentCoroutineContext()
            val generation = inference.generate(AgentInferenceRequest(prompts, finalOnly, thinkingAllowed, outputLimit)) { delta ->
                context.ensureActive()
                if (!consent.isCurrent(grant.revision)) throw CancellationException("Agent consent changed")
                if (buffer.length < AgentOutputParser.MAX_OUTPUT_LENGTH) {
                    buffer.append(delta.take(AgentOutputParser.MAX_OUTPUT_LENGTH - buffer.length))
                }
                val raw = buffer.toString()
                val snapshot = AgentStreamDecoder.decode(raw, thinkingAllowed)
                val visible = AgentOutputParser.visibleAnswer(snapshot.answer)
                state = state.copy(answer = visible.orEmpty(), thinking = (thoughts + snapshot.thinking)
                    .filter { it.isNotBlank() }.joinToString("\n\n"))
                if (pendingId == null && (visible.orEmpty() != snapshot.answer.trim() || raw.endsWith("<tool_call>"))) {
                    pendingId = addStep(AgentStepKind.PREPARING, "preparing_tool", "")
                } else {
                    if (visible != null && visible == snapshot.answer.trim()) {
                        pendingId?.let { id -> steps.removeAll { it.id == id } }
                        pendingId = null
                    }
                    publish()
                }
            }
            checkActive()
            require(generation.text.length <= AgentOutputParser.MAX_OUTPUT_LENGTH) { "Model output exceeds agent limit" }
            state = state.copy(metrics = generation.metrics)
            pendingId?.let { id -> steps.removeAll { it.id == id } }
            if (generation.stopped) throw CancellationException("Generation stopped")
            if (generation.metrics["error"] == true || generation.metrics["session_invalid"] == true || generation.metrics["success"] == false) {
                throw AgentInferenceException(generation.metrics)
            }
            check(!generation.text.startsWith("Error:")) { "Inference failed" }
            val snapshot = AgentStreamDecoder.decode(generation.text, thinkingAllowed)
            if (snapshot.thinking.isNotBlank()) thoughts.add(snapshot.thinking)
            state = state.copy(thinking = thoughts.joinToString("\n\n"))
            val rawTool = if (snapshot.unclosedThinking) AgentOutputParser.parseToolCallTags(generation.text) else null
            if (snapshot.answer.isEmpty() && snapshot.unclosedThinking && rawTool == null && thinkingAllowed && !retriedThinking) {
                retriedThinking = true
                thinkingAllowed = false
                finalOnly = finalOnly || exchanges.isNotEmpty()
                publish()
                continue
            }
            val output = snapshot.answer
            val parsed = rawTool ?: AgentOutputParser.parse(output)
            val observations = mutableListOf<String>()
            if (!finalOnly) {
                for (memory in parsed?.memoryUpdates.orEmpty()) observations.add(persistMemory(memory.content, memory.category))
                for (skill in parsed?.skillUpdates.orEmpty()) observations.add(persistSkill(skill))
            }
            val calls = if (finalOnly) emptyList() else parsed?.calls.orEmpty()
            if (calls.isEmpty()) {
                val plain = parsed?.reply ?: if (parsed == null || finalOnly) AgentOutputParser.finalAnswerText(output) else ""
                state = state.copy(answer = plain)
                if (observations.any { it.startsWith("[TOOL_ERROR]") } && !finalOnly) {
                    exchanges.add(Exchange(plain.ifEmpty { "Storage updates requested" }, observations.toList(), 0))
                    finalOnly = true
                    continue
                }
                break
            }
            state = state.copy(answer = "")
            toolRounds++
            for (rawCall in calls) {
                checkActive()
                var call = try { AgentCallValidator.validate(rawCall) } catch (_: IllegalArgumentException) {
                    addStep(kind(rawCall.tool), rawCall.type.take(64), "", AgentStepStatus.FAILED, "Invalid tool or parameters")
                    observations.add("[TOOL_ERROR] Invalid tool or parameters. Use only the provided schemas")
                    continue
                }
                if (call.tool == AgentTool.MEMORY) {
                    observations.add(persistMemory(call.value, "preference"))
                    continue
                }
                if (call.tool == AgentTool.SEARCH) call = call.copy(value = AgentInputPolicy.privateSearchQuery(call.value, request.userText))
                if (call.tool == AgentTool.READ_PAGE) {
                    val number = AgentInputPolicy.resultNumber(call.value)
                    val rawUrl = if (number != null) resultUrls.getOrNull(number - 1) else call.value
                    val url = try { rawUrl?.let { urlPolicy.validate(it).toString() } } catch (_: IllegalArgumentException) { null }
                    if (url == null) {
                        addStep(kind(call.tool), call.tool.wireName, call.value, AgentStepStatus.SKIPPED, "Invalid or non-public URL")
                        observations.add("[TOOL_ERROR] Use a public HTTP(S) URL or a valid search result number")
                        continue
                    }
                    if (url !in allowedUrls) {
                        if (searched) {
                            addStep(kind(call.tool), call.tool.wireName, url, AgentStepStatus.SKIPPED, "URL was not supplied by the user or search")
                            observations.add("[TOOL_ERROR] This URL is not authorized. Choose a numbered search result or answer from existing results")
                            continue
                        }
                        call = ValidatedAgentCall(AgentTool.SEARCH, AgentInputPolicy.coreSearchQuery(request.userText))
                    } else call = call.copy(value = url)
                }
                val id = addStep(kind(call.tool), call.tool.wireName, call.value)
                if (call.tool in setOf(AgentTool.SEARCH, AgentTool.READ_PAGE) && !consent.canUseNetwork(grant.revision)) {
                    finishStep(id, AgentStepStatus.SKIPPED, "Network consent is required")
                    observations.add("[TOOL_ERROR] Network is not permitted")
                    continue
                }
                val keyValue = when (call.tool) {
                    AgentTool.SEARCH -> call.value.lowercase(Locale.ROOT)
                    AgentTool.CALCULATOR -> AgentCalculator.normalize(call.value).replace(Regex("\\s+"), "").lowercase(Locale.ROOT)
                    else -> call.value
                }
                val key = "${call.tool.wireName}:$keyValue"
                if (key in seen) {
                    finishStep(id, AgentStepStatus.SKIPPED, "Duplicate call")
                    observations.add("[TOOL_ERROR] Duplicate call skipped; use the earlier result")
                    continue
                }
                if (state.toolCalls >= budget.maxToolCalls || (call.tool == AgentTool.READ_PAGE && state.pageReads >= budget.maxPageReads)) {
                    finishStep(id, AgentStepStatus.SKIPPED, "Tool budget exhausted")
                    observations.add("[TOOL_ERROR] Tool budget exhausted")
                    continue
                }
                seen.add(key)
                state = state.copy(toolCalls = state.toolCalls + 1,
                    pageReads = state.pageReads + if (call.tool == AgentTool.READ_PAGE) 1 else 0)
                publish()
                val result = try {
                    withTimeoutOrNull(35000) { tools.execute(call, AgentToolContext(request.userText, grant.revision)) }
                        ?: AgentToolResult.error("Tool execution timed out")
                } catch (e: CancellationException) { throw e }
                catch (_: Exception) { AgentToolResult.error("Tool execution failed") }
                checkActive()
                finishStep(id, if (result.success) AgentStepStatus.DONE else AgentStepStatus.FAILED, summary(result))
                if (call.tool == AgentTool.SEARCH) {
                    searched = true
                    if (result.success) {
                        resultUrls = result.searchHits.take(5).mapNotNull { hit ->
                            try { urlPolicy.validate(hit.url).toString() } catch (_: IllegalArgumentException) { null }
                        }
                        allowedUrls.addAll(resultUrls)
                    }
                }
                observations.add(result.observation.take(4000))
            }
            checkActive()
            exchanges.add(Exchange(AgentPrompts.callsMessage(calls), observations.toList(), budget.maxToolCalls - state.toolCalls))
            if (toolRounds >= budget.maxPasses || state.toolCalls >= budget.maxToolCalls ||
                observations.isEmpty() || observations.all { it.startsWith("[TOOL_ERROR]") }) finalOnly = true
        }
        checkActive()
        if (!memorySaved && consent.canPersist(grant.revision)) {
            AgentInputPolicy.explicitMemory(request.userText)?.let { persistMemory(it, "preference") }
        }
        state = state.copy(phase = AgentPhase.COMPLETED, answer = state.answer
            .ifBlank { "No answer was generated. Please try again with a more specific request." })
        publish()
        return state
    }

    private suspend fun countTokens(messages: List<AgentMessage>): Pair<Int, Boolean> {
        currentCoroutineContext().ensureActive()
        val count = inference.countTokens(messages)
        if (count != null) { require(count >= 0); return count to false }
        val estimate = messages.sumOf { (it.content.toByteArray(StandardCharsets.UTF_8).size.toLong() + 1) / 2 + 32 }
        return estimate.coerceAtMost(Int.MAX_VALUE.toLong()).toInt() to true
    }

    private fun kind(tool: AgentTool?): AgentStepKind = when (tool) {
        AgentTool.TIME -> AgentStepKind.TIME
        AgentTool.CALCULATOR -> AgentStepKind.CALCULATOR
        AgentTool.SEARCH -> AgentStepKind.SEARCH
        AgentTool.READ_PAGE -> AgentStepKind.READ_PAGE
        AgentTool.MEMORY -> AgentStepKind.MEMORY
        null -> AgentStepKind.OTHER
    }

    private fun summary(result: AgentToolResult): String = when {
        !result.success -> result.observation.take(160)
        result.searchHits.isNotEmpty() -> "${result.searchHits.size} search results"
        result.observation.startsWith("[CALC_RESULT]") -> "= " + result.observation.substringAfter("result: ")
        result.observation.startsWith("[TIME_RESULT]") -> result.observation.substringAfter("datetime: ").substringBefore('\n')
        result.observation.startsWith("[BROWSE_RESULT]") -> result.observation.substringAfter("title: ").substringBefore('\n').ifBlank { "Page read" }
        else -> result.observation.take(160)
    }
}
