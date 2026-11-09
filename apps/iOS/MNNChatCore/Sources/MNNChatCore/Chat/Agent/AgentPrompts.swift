import Foundation

/// Limits for one agent request, sized for on-device models.
struct AgentBudget {
    /// Tool rounds before the model must answer. Small models tend to keep
    /// re-searching instead of answering, and each round costs seconds.
    var maxPasses = 3
    var maxToolCalls = 6
    var maxPageReads = 4
}

enum AgentPrompts {
    /// Tool schemas in the OpenAI function format that chat templates render.
    static let toolSchemas: [String] = [
        #"{"type":"function","function":{"name":"web_search","description":"Search the web. Returns result titles, snippets and URLs.","parameters":{"type":"object","properties":{"query":{"type":"string","description":"Search keywords"}},"required":["query"]}}}"#,
        #"{"type":"function","function":{"name":"browser_url","description":"Read the text of a web page from the search results.","parameters":{"type":"object","properties":{"url":{"type":"string","description":"The number of a search result (for example 2), or a URL copied from the search results"}},"required":["url"]}}}"#,
        #"{"type":"function","function":{"name":"get_current_time","description":"Get the exact local date, time and weekday.","parameters":{"type":"object","properties":{}}}}"#,
        #"{"type":"function","function":{"name":"calculator","description":"Evaluate an arithmetic expression exactly. Supports + - * / % ^, parentheses, sqrt, round, abs.","parameters":{"type":"object","properties":{"expression":{"type":"string","description":"For example 225.83*7.12"}},"required":["expression"]}}}"#,
        #"{"type":"function","function":{"name":"save_memory","description":"Save a lasting fact or preference about the user for future conversations.","parameters":{"type":"object","properties":{"content":{"type":"string","description":"The fact or preference, as a short sentence"}},"required":["content"]}}}"#,
    ]

    /// The tool section mirrors the one Qwen3.5's chat template renders for
    /// native tool calls (the wrapper cannot pass `tools` to the template),
    /// since small models follow the format they were trained on far more
    /// reliably than a custom one. Other models get the same clear contract.
    static func systemPrompt(memories: [AgentMemory], skills: [AgentSkill], now: Date = Date()) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyy-MM-dd EEEE"
        var lines = memoryBlock(memories) + [
            "# Tools",
            "",
            "You have access to the following functions:",
            "",
            "<tools>",
        ] + toolSchemas + [
            "</tools>",
            "",
            "If you choose to call a function ONLY reply in the following format with NO suffix:",
            "",
            "<tool_call>",
            "<function=example_function_name>",
            "<parameter=example_parameter_1>",
            "value_1",
            "</parameter>",
            "</function>",
            "</tool_call>",
            "",
            "<IMPORTANT>",
            "Reminder:",
            "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags",
            "- Required parameters MUST be specified",
            "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls",
            "</IMPORTANT>",
            "",
            "You are MNN Chat's assistant, running on the user's phone.",
            "Today's date is \(formatter.string(from: now)). For the current time, call get_current_time.",
            "Call web_search for anything about today or recent events (weather, news, prices, scores), for facts you are not sure about, and whenever the user asks you to search or look something up. Do not guess such information.",
            "For any calculation beyond trivial whole-number arithmetic (decimals, percentages, currency or unit conversions), call calculator instead of computing in your head, and use its result exactly.",
            "Do not call functions for today's date or weekday, or common knowledge you are sure about; answer those directly.",
            "After you receive <tool_response>, answer in the user's language based on it, or call another function if needed.",
            "When the user asks you to remember something, or states a lasting preference or fact about themselves, call save_memory, then reply briefly.",
        ]
        if !skills.isEmpty {
            lines.append("")
            lines.append("Saved procedures that match this request:")
            for skill in skills.prefix(3) { lines.append("- \(skill.name): \(skill.actionTemplate)") }
        }
        return lines.joined(separator: "\n")
    }

    /// For the closing answer: no tool definitions, so the model cannot keep
    /// requesting tools instead of answering.
    static func answerSystemPrompt(memories: [AgentMemory], now: Date = Date()) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyy-MM-dd EEEE"
        var lines = memoryBlock(memories) + [
            "You are MNN Chat's assistant, running on the user's phone. Today's date is \(formatter.string(from: now)).",
            "Answer the user's question in the user's language, using the <tool_response> results in this conversation.",
            "If the results do not contain the answer, say so briefly and give your best answer.",
        ]
        return lines.joined(separator: "\n")
    }

    /// Saved facts, first in the system prompt so small models do not lose
    /// them behind the tool section. First-person notes ("我是…") are
    /// rewritten in the third person so they read as facts about the user.
    static func memoryBlock(_ memories: [AgentMemory]) -> [String] {
        guard !memories.isEmpty else { return [] }
        var lines = ["# About the user (always take this into account in your answers)"]
        var used = 0
        for memory in memories {
            used += memory.content.count
            if used > 1200 { break }
            lines.append("- " + thirdPerson(memory.content))
        }
        return lines + [""]
    }

    /// Small models follow notes in the user turn far more than ones in
    /// the system prompt, so memories are also appended to the request
    /// sent to the model (never shown in the chat).
    static func requestWithMemories(_ request: String, memories: [AgentMemory]) -> String {
        guard !memories.isEmpty else { return request }
        let notes = memories.prefix(8).map { thirdPerson($0.content) }.joined(separator: "；")
        return request + "\n\n（回答时请参考已知的用户信息：\(notes)）"
    }

    static func thirdPerson(_ text: String) -> String {
        var result = text
        for (first, third) in [("我的", "用户的"), ("我是", "用户是"), ("我喜欢", "用户喜欢"), ("我", "用户")] {
            result = result.replacingOccurrences(of: first, with: third)
        }
        if result.lowercased().hasPrefix("i ") { result = "The user " + result.dropFirst(2) }
        return result
    }

    /// Skills whose trigger keywords appear in the request.
    static func matchingSkills(_ skills: [AgentSkill], for request: String) -> [AgentSkill] {
        let text = request.lowercased()
        return skills.filter { skill in skill.triggers.contains { !$0.isEmpty && text.contains($0.lowercased()) } }
    }

    /// Results are wrapped the way chat templates render the `tool` role.
    static func toolResultsMessage(_ observations: [String], remainingCalls: Int) -> String {
        var message = observations.map { "<tool_response>\n\($0)\n</tool_response>" }.joined(separator: "\n")
        message += remainingCalls > 0
            ? "\nAnswer the user's question now in the user's language based on these results, or call another function only if it is really needed. If the answer needs a conversion or calculation, call calculator first."
            : "\nNo more function calls are available. Answer the user's question now in the user's language based on these results."
        return message
    }

    static let finalAnswerMessage = "No more function calls are available. Answer the user's question now in the "
        + "user's language, using the tool responses above."
}
