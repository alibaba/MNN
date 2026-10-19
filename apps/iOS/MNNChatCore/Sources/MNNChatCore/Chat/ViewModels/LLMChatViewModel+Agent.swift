import CMNNInference
import ExyteChat
import Foundation

/// Agent mode: a multi-pass loop where the local model may request tools
/// (web search, page reading, current time) as JSON, the app runs them, and
/// the results are fed back until the model answers. Adapted from the
/// Android implementation in alibaba/MNN#4512.
extension LLMChatViewModel {
    private struct PassResult {
        var text = ""
        var stopped = false
        var performance: String?
    }

    /// Answers `request` into the empty assistant message at the end of the
    /// conversation. The caller has already set `isProcessing`.
    ///
    /// The model's own thinking (when thinking mode is on) goes to the
    /// thinking panel; tool calls are recorded as `toolSteps` and shown in a
    /// separate tool panel.
    @MainActor
    func runAgentRequest(_ request: String, generation: UInt64) async {
        guard let engine = llm, generation == contextGeneration, isStarted,
              let messageId = messages.last(where: { $0.user.id == "2" })?.id else {
            if activeRequestGeneration == generation { finishRequest() }
            return
        }
        await llmState.setProcessing(true)
        isProcessing = true
        contextNotice = nil
        readsNativeContextUsage = false
        currentStreamingMessageId = messageId
        let stateManager = StreamingMessageStateManager(messageId: messageId)
        streamingStates[messageId] = stateManager
        stateManager.startStreaming()

        let store = AgentStore.shared
        let skills = AgentPrompts.matchingSkills(store?.allSkills() ?? [], for: request)
        let memories = store?.allMemories() ?? []
        let toolSystem = AgentPrompts.systemPrompt(memories: memories, skills: skills)
        let answerSystem = AgentPrompts.answerSystemPrompt(memories: memories)
        // What the model sees as the user's question; `request` stays the
        // user's own words for search refinement and memory detection.
        let modelRequest = AgentPrompts.requestWithMemories(request, memories: memories)
        var history = isConversationContextEnabled
            ? ConversationContextBuilder.turns(from: Array(messages.dropLast(2))).flatMap(\.prompts) : []
        // Tool rounds of this request: assistant tool call, then user tool results.
        var exchange: [[String: String]] = []
        var thoughts: [String] = []     // thinking of finished passes
        var liveThinking = ""           // thinking of the running pass
        var steps: [ToolStep] = []
        var answer = ""
        var performance: String?
        var stopped = false
        var failure: String?
        let budget = AgentBudget()
        var toolCalls = 0, pageReads = 0
        var seen = Set<String>()
        // Sites the model may read: those in search results or written by
        // the user. Small models otherwise invent URLs that do not exist.
        var knownHosts = Self.hosts(in: request + "\n" + history.compactMap { $0["user"] }.joined(separator: "\n"))
        var searched = false
        var resultURLs: [String] = []   // numbered URLs of the latest search
        // Qwen3.5-style templates pre-open the think block, so its stream
        // carries only the closing tag; others emit a literal <think>.
        let thinkingOn = supportsThinkingMode && isThinkingModeEnabled
        var expectsThinking = thinkingOn && interactor.thinkingExpectedByConfig
        var thinkingActive = thinkingOn
        func stopThinking() {
            guard thinkingActive else { return }
            thinkingActive = false
            expectsThinking = false
            engine.setThinkingModeEnabled(false)
        }

        func show() {
            let thinking = (thoughts + [liveThinking]).filter { !$0.isEmpty }.joined(separator: "\n\n")
            interactor.setMessage(messageId, text: answer, thinkText: thinking, toolSteps: steps)
        }
        func isCurrent() -> Bool { generation == contextGeneration && isStarted }

        engine.setContextTokenLimit(contextLimitTokens)
        engine.setConversationContextEnabled(true)
        engine.setEnableAudioOutput(false)

        var finalOnly = false
        // +2: one final-answer pass and one possible retry without thinking.
        for pass in 1...(budget.maxPasses + 2) {
            guard isCurrent() else { stopped = true; break }
            // The closing answer is written without thinking: small models
            // otherwise tend to spend it all on thinking and never answer.
            if finalOnly { stopThinking() }
            // The last user message is passed as the input; the rest as prompts.
            let lastMessage = finalOnly ? AgentPrompts.finalAnswerMessage : (exchange.last?["user"] ?? modelRequest)
            func buildPrompts() -> [[String: String]] {
                var prompts = [["system": finalOnly ? answerSystem : toolSystem]] + history + [["user": modelRequest]] + exchange
                if finalOnly { prompts.append(["user": AgentPrompts.finalAnswerMessage]) }
                prompts.removeLast()
                return prompts
            }
            var prompts = buildPrompts()
            // Keep within the context budget: drop the oldest turns first.
            let outputReserve = min(modelConfigManager.readMaxNewTokens(), contextLimitTokens / 2)
            while !history.isEmpty, await countTokens(prompts + [["user": lastMessage]], engine: engine) > contextLimitTokens - outputReserve {
                history.removeFirst(min(2, history.count))
                prompts = buildPrompts()
            }

            let splitter = ThinkResultProcessor()
            splitter.expectThinking = expectsThinking
            var pendingIndex: Int?
            let result = await generatePass(engine: engine, prompts: prompts, input: lastMessage) { delta in
                splitter.process(delta: delta)
                liveThinking = splitter.thinking
                // Stream a plain-text answer as it is written; while a tool
                // call is being written, show a pending tool step instead.
                if let visible = AgentOutputParser.visibleAnswer(splitter.answer) {
                    answer = visible
                } else if splitter.answer.contains("<tool_call") || splitter.answer.hasPrefix("{"), pendingIndex == nil {
                    answer = ""
                    steps.append(ToolStep(kind: .other, title: AppText.text("agent.step.preparing"), detail: ""))
                    pendingIndex = steps.count - 1
                }
                show()
            }
            if let pendingIndex { steps.remove(at: pendingIndex) }
            if !liveThinking.isEmpty { thoughts.append(liveThinking) }
            liveThinking = ""
            performance = result.performance ?? performance
            if result.stopped || !isCurrent() { stopped = true; break }
            if result.text.hasPrefix("Error:") {
                failure = result.text.hasPrefix("Error: Input exceeds the context budget.")
                    ? AppText.text("context.error.tooLong") : result.text
                break
            }

            // Small models sometimes stop (or run out of tokens) before closing
            // their thinking. Such a pass has no answer: run any tool call it
            // wrote, otherwise ask again with thinking off. Never show the
            // thinking text as the answer.
            let unclosedThinking = splitter.answer.isEmpty && !splitter.thinking.isEmpty
            if unclosedThinking, AgentOutputParser.parseToolCallTags(result.text) == nil, thinkingActive {
                stopThinking()
                finalOnly = finalOnly || !exchange.isEmpty
                continue
            }
            let output = unclosedThinking ? result.text : splitter.answer
            let parsed = AgentOutputParser.parse(output)
            for memory in parsed?.memoryUpdates ?? [] { store?.addMemory(memory.content, category: memory.category) }
            for skill in parsed?.skillUpdates ?? [] { store?.saveSkill(skill) }
            let calls = finalOnly ? [] : (parsed?.calls ?? [])
            if calls.isEmpty {
                answer = unclosedThinking ? ""
                    : parsed?.reply ?? (parsed == nil ? output.trimmingCharacters(in: .whitespacesAndNewlines) : "")
                // A final pass that still wrote a tool call: keep any prose before it.
                if answer.isEmpty, let prose = AgentOutputParser.visibleAnswer(output.components(separatedBy: "<tool_call").first ?? "") {
                    answer = prose
                }
                break
            }

            // Run the requested tools, skipping duplicates and over-budget calls.
            answer = ""
            var observations: [String] = []
            for var call in calls {
                guard isCurrent() else { break }
                // A search without keywords searches the user's question.
                if call.tool == "web_search" {
                    let query = (call.query ?? "").trimmingCharacters(in: .whitespaces)
                    call.query = query.isEmpty ? AgentTools.coreText(of: request)
                        : AgentTools.refinedQuery(query, request: request)
                }
                // Memory is saved locally; it needs no network or budget.
                if call.tool == "save_memory" {
                    let content = (call.content ?? call.query ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
                    var step = ToolStep(kind: .memory, title: Self.title(of: call), detail: content)
                    if content.isEmpty || seen.contains("memory:" + content) {
                        step.status = .skipped
                        step.result = AppText.text("agent.result.duplicate")
                    } else {
                        seen.insert("memory:" + content)
                        store?.addMemory(content, category: "preference")
                        step.status = .done
                        step.result = AppText.text("agent.result.saved")
                        observations.append("[MEMORY_SAVED] \(content)")
                    }
                    steps.append(step); show()
                    continue
                }
                // "2", "[2]" or "结果2": a search result number.
                if call.tool == "browser_url", let number = Self.resultNumber(call.url ?? call.query ?? ""),
                   resultURLs.indices.contains(number - 1) {
                    call.url = resultURLs[number - 1]
                }
                if call.tool == "browser_url", !Self.isKnown(call.url ?? call.query ?? "", hosts: knownHosts) {
                    if searched {
                        let url = call.url ?? call.query ?? ""
                        steps.append(ToolStep(kind: .browse, title: Self.title(of: call), detail: url,
                                              status: .skipped, result: AppText.text("agent.result.unknownURL")))
                        show()
                        let choices = resultURLs.enumerated().map { "\($0.offset + 1). \($0.element)" }.joined(separator: "\n")
                        observations.append("[TOOL_ERROR] \(url) is not in the search results, so it was not opened. "
                            + "To read a page, call browser_url with the number of one of these results:\n\(choices)\n"
                            + "Or answer from the search results.")
                        continue
                    }
                    // Nothing searched yet: search the question instead of guessing a page.
                    call = AgentCall(type: "web_search", query: request)
                }
                let detail = call.query ?? call.url ?? ""
                var step = ToolStep(kind: Self.kind(of: call.tool), title: Self.title(of: call), detail: detail)
                let key = call.tool + ":" + detail.lowercased()
                if seen.contains(key) {
                    step.status = .skipped
                    step.result = AppText.text("agent.result.duplicate")
                    steps.append(step); show(); continue
                }
                if toolCalls >= budget.maxToolCalls || (call.tool == "browser_url" && pageReads >= budget.maxPageReads) {
                    step.status = .skipped
                    step.result = AppText.text("agent.result.budget")
                    steps.append(step); show(); continue
                }
                seen.insert(key)
                toolCalls += 1
                if call.tool == "browser_url" { pageReads += 1 }
                steps.append(step)
                let index = steps.count - 1
                show()
                let observation = await AgentTools.execute(call)
                guard isCurrent() else { break }
                let failed = observation.hasPrefix("[TOOL_ERROR]") || observation.hasPrefix("[SEARCH_ERROR]")
                    || observation.hasPrefix("[CALC_ERROR]")
                    || observation.hasPrefix("[BROWSE_ERROR]")
                steps[index].status = failed ? .failed : .done
                steps[index].result = Self.summary(of: observation)
                if call.tool == "web_search" {
                    searched = true
                    knownHosts.formUnion(Self.hosts(in: observation))
                    let urls = observation.components(separatedBy: "\n").compactMap { line -> String? in
                        let trimmed = line.trimmingCharacters(in: .whitespaces)
                        return trimmed.hasPrefix("url: ") ? String(trimmed.dropFirst(5)) : nil
                    }
                    if !urls.isEmpty { resultURLs = urls }
                }
                show()
                observations.append(observation)
            }
            guard isCurrent() else { stopped = true; break }
            exchange.append(["assistant": output])
            if observations.isEmpty {
                // Every call was a duplicate or over budget: ask for the answer.
                finalOnly = true
                continue
            }
            exchange.append(["user": AgentPrompts.toolResultsMessage(observations,
                                                                    remainingCalls: budget.maxToolCalls - toolCalls)])
            if pass == budget.maxPasses { finalOnly = true }
        }

        // Small models often acknowledge "remember …" without calling
        // save_memory. An explicit request is saved by the app instead.
        if !stopped, !steps.contains(where: { $0.kind == .memory }), let memory = Self.explicitMemory(in: request) {
            store?.addMemory(memory, category: "preference")
            steps.append(ToolStep(kind: .memory, title: AppText.text("agent.tool.memory"), detail: memory,
                                  status: .done, result: AppText.text("agent.result.saved")))
        }

        // Restore the normal-chat engine settings.
        configureThinkingMode()
        engine.setEnableAudioOutput(modelConfigManager.readEnableAudioOutput())

        for index in steps.indices where steps[index].status == .running { steps[index].status = .failed }
        if !answer.isEmpty { answer = AgentTools.correctArithmetic(in: answer) }
        if let failure, answer.isEmpty { answer = failure }
        if answer.isEmpty, !stopped { answer = AppText.text("agent.noAnswer") }
        show()
        if let performance { interactor.updatePerformanceData(performance, for: messageId) }
        stateManager.markOutputComplete()
        streamingStates[messageId] = stateManager
        NotificationCenter.default.post(name: NSNotification.Name("StreamingOutputCompleted"), object: nil,
                                        userInfo: ["messageId": messageId])
        DispatchQueue.main.asyncAfter(deadline: .now() + min(max(Double(answer.count) * 0.02 + 2, 5), 30)) { [weak self] in
            self?.finishStreamingPresentation(for: messageId)
        }
        readsNativeContextUsage = true
        refreshContextUsage()
        if stopped { contextNotice = AppText.text("chat.generationStopped") }
        await llmState.setProcessing(false)
        if activeRequestGeneration == generation { finishRequest() }
    }

    /// One model pass with explicit prompts; returns the complete output.
    @MainActor
    private func generatePass(engine: LLMInferenceEngineWrapper, prompts: [[String: String]], input: String,
                              onDelta: @escaping (String) -> Void) async -> PassResult {
        engine.addPrompts(from: prompts as [[AnyHashable: Any]])
        let result: PassResult = await withCheckedContinuation { continuation in
            var result = PassResult(), completed = false
            engine.processInput(input, withOutput: { output in
                guard !completed else { return }
                if output.hasPrefix("<performance>") {
                    result.performance = String(output.dropFirst("<performance>".count))
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                } else if output == "<eop>" || output == "<stoped>" {
                    completed = true
                    result.stopped = output == "<stoped>"
                    continuation.resume(returning: result)
                } else {
                    result.text += output
                    onDelta(output)
                }
            }, showPerformance: true)
        }
        while engine.isProcessing() { try? await Task.sleep(for: .milliseconds(10)) }
        return result
    }

    @MainActor
    private func countTokens(_ prompts: [[String: String]], engine: LLMInferenceEngineWrapper) async -> Int {
        await withCheckedContinuation { continuation in
            engine.countTokens(for: prompts as [[AnyHashable: Any]]) { continuation.resume(returning: max($0, 0)) }
        }
    }

    private static func kind(of tool: String) -> ToolStep.Kind {
        switch tool {
        case "web_search": .search
        case "browser_url": .browse
        case "get_current_time": .time
        case "save_memory": .memory
        case "calculator": .calculate
        default: .other
        }
    }

    private static func title(of call: AgentCall) -> String {
        switch call.tool {
        case "web_search": AppText.text("agent.tool.search")
        case "browser_url": AppText.text("agent.tool.browse")
        case "get_current_time": AppText.text("agent.tool.time")
        case "save_memory": AppText.text("agent.tool.memory")
        case "calculator": AppText.text("agent.tool.calculator")
        default: call.type
        }
    }

    /// A short, user-facing summary of a tool observation.
    private static func summary(of observation: String) -> String {
        if observation.hasPrefix("[SEARCH_RESULT]") {
            let count = observation.components(separatedBy: "\n").filter { $0.range(of: #"^\d+\. "#, options: .regularExpression) != nil }.count
            return String(format: AppText.text("agent.summary.search"), count)
        }
        if observation.hasPrefix("[BROWSE_RESULT]") {
            let title = observation.components(separatedBy: "\n").first { $0.hasPrefix("title: ") }?.dropFirst(7)
            return title.map { String($0.prefix(40)) } ?? AppText.text("agent.summary.page")
        }
        if observation.hasPrefix("[CALC_RESULT]") {
            return observation.components(separatedBy: "\n").first { $0.hasPrefix("result: ") }.map { "= " + $0.dropFirst(8) } ?? ""
        }
        if observation.hasPrefix("[CALC_ERROR]") { return AppText.text("agent.result.calcError") }
        if observation.hasPrefix("[TIME_RESULT]") {
            return observation.components(separatedBy: "\n").first { $0.hasPrefix("datetime: ") }.map { String($0.dropFirst(10)) } ?? ""
        }
        if let status = observation.range(of: #"HTTP (\d{3})"#, options: .regularExpression) {
            return String(format: AppText.text("agent.result.httpError"), String(observation[status].dropFirst(5)))
        }
        if observation.hasPrefix("[SEARCH_ERROR]") { return AppText.text("agent.result.noResults") }
        if observation.hasPrefix("[BROWSE_ERROR]") || observation.contains("failed:") { return AppText.text("agent.result.unreachable") }
        let detail = observation.components(separatedBy: "] ").dropFirst().joined(separator: "] ")
        return String(detail.prefix(80))
    }

    /// The content of an explicit memory request ("记住：…", "以后…",
    /// "remember that …"), or nil.
    static func explicitMemory(in request: String) -> String? {
        let text = request.trimmingCharacters(in: .whitespacesAndNewlines)
        let lower = text.lowercased()
        let markers = ["请记住", "记住", "记一下", "请记得", "记得", "以后", "今后", "从现在起", "remember", "from now on"]
        guard markers.contains(where: { lower.contains($0) }) else { return nil }
        var content = text
        for prefix in ["请记住", "记住", "记一下", "请记得", "记得", "remember that", "remember"] {
            if content.lowercased().hasPrefix(prefix) { content = String(content.dropFirst(prefix.count)); break }
        }
        content = content.trimmingCharacters(in: CharacterSet(charactersIn: " ：:，,。.").union(.whitespacesAndNewlines))
        guard content.count >= 2 else { return nil }
        return String(content.prefix(AgentStore.maxMemoryLength))
    }

    static func resultNumber(_ text: String) -> Int? {
        let digits = text.trimmingCharacters(in: CharacterSet(charactersIn: " []()#结果第条.。"))
        return Int(digits)
    }

    /// Hosts of the http(s) URLs in `text`, without a leading "www.".
    static func hosts(in text: String) -> Set<String> {
        guard let detector = try? NSDataDetector(types: NSTextCheckingResult.CheckingType.link.rawValue) else { return [] }
        let matches = detector.matches(in: text, range: NSRange(text.startIndex..., in: text))
        return Set(matches.compactMap { $0.url?.host?.lowercased() }.map(normalizedHost))
    }

    static func isKnown(_ url: String, hosts: Set<String>) -> Bool {
        let withScheme = url.lowercased().hasPrefix("http") ? url : "https://" + url
        guard let host = URL(string: withScheme)?.host?.lowercased() else { return false }
        return hosts.contains(normalizedHost(host))
    }

    private static func normalizedHost(_ host: String) -> String {
        host.hasPrefix("www.") ? String(host.dropFirst(4)) : host
    }
}
