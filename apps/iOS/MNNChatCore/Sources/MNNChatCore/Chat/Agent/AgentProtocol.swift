import Foundation

/// One tool request from the model (an entry of `system_calls`).
struct AgentCall: Equatable {
    var type: String
    var query: String? = nil
    var url: String? = nil
    /// For save_memory.
    var content: String? = nil

    /// Normalized tool name; aliases used by models are folded together.
    var tool: String {
        switch type.trimmingCharacters(in: .whitespaces).lowercased() {
        case "browser_url", "browse_url", "web_browse", "open_url", "read_url": "browser_url"
        case "web_search", "search": "web_search"
        case "get_current_time", "current_time", "time": "get_current_time"
        case "save_memory", "remember", "memory_update", "add_memory": "save_memory"
        case "calculator", "calculate", "calc", "math", "compute": "calculator"
        case let other: other
        }
    }
}

/// A parsed model turn: a final reply, tool requests, or both.
struct AgentResponse: Equatable {
    var reply: String?
    var calls: [AgentCall] = []
    var memoryUpdates: [(content: String, category: String)] = []
    var skillUpdates: [AgentSkill] = []

    var isEmpty: Bool { (reply ?? "").isEmpty && calls.isEmpty && memoryUpdates.isEmpty && skillUpdates.isEmpty }

    static func == (lhs: AgentResponse, rhs: AgentResponse) -> Bool {
        lhs.reply == rhs.reply && lhs.calls == rhs.calls && lhs.skillUpdates == rhs.skillUpdates
            && lhs.memoryUpdates.map(\.content) == rhs.memoryUpdates.map(\.content)
    }
}

/// Tolerant parser for agent output. Small on-device models format JSON
/// imperfectly, so it accepts: a strict object, a fenced block, an object
/// embedded in prose, a bare tool object, a bare array of tools,
/// `system_calls` as an object or a JSON string, and agent JSON nested in
/// `reply`. Ported from the Android implementation (PR #4512).
enum AgentOutputParser {
    static let toolTypes: Set<String> = ["get_current_time", "web_search", "browser_url",
                                         "browse_url", "web_browse", "open_url", "read_url", "search",
                                         "current_time", "time", "save_memory", "remember",
                                         "memory_update", "add_memory", "calculator", "calculate",
                                         "calc", "math", "compute"]

    /// Returns nil when the text contains no agent JSON (a plain final answer).
    static func parse(_ text: String) -> AgentResponse? {
        let cleaned = stripThinking(text)
        if let tagged = parseToolCallTags(cleaned) { return tagged }
        for candidate in candidates(cleaned) {
            guard var response = parseCandidate(candidate) else { continue }
            if response.calls.isEmpty, let reply = response.reply, let nested = parseNested(reply) {
                response = nested
            }
            if !response.isEmpty {
                // A JSON block that only carries memory/skill updates after a
                // prose answer: the prose is the reply.
                if (response.reply ?? "").isEmpty, response.calls.isEmpty, candidate != cleaned {
                    let prose = cleaned.replacingOccurrences(of: candidate, with: "")
                        .replacingOccurrences(of: "```json", with: "").replacingOccurrences(of: "```", with: "")
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                    if !prose.isEmpty { response.reply = prose }
                }
                return response
            }
        }
        return nil
    }

    /// Removes `<think>…</think>` blocks (and an unterminated leading one).
    static func stripThinking(_ text: String) -> String {
        var result = text
        while let open = result.range(of: "<think>") {
            if let close = result.range(of: "</think>", range: open.upperBound..<result.endIndex) {
                result.removeSubrange(open.lowerBound..<close.upperBound)
            } else {
                result.removeSubrange(open.lowerBound..<result.endIndex)
            }
        }
        if let close = result.range(of: "</think>") { result.removeSubrange(result.startIndex..<close.upperBound) }
        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// The part of a streamed output that may be shown as the answer: nil
    /// while it looks like it is becoming a tool call.
    static func visibleAnswer(_ partial: String) -> String? {
        var text = stripThinking(partial)
        if let tag = text.range(of: "<tool_call") { text = String(text[..<tag.lowerBound]) }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty || trimmed.hasPrefix("{") || trimmed.hasPrefix("[") || trimmed.hasPrefix("`")
            || "<tool_call>".hasPrefix(trimmed) || partial.contains("<tool_call") { return nil }
        return trimmed
    }

    /// Native `<tool_call>` blocks: Qwen3.5's XML form
    /// (`<function=name><parameter=key>value</parameter></function>`) and the
    /// older JSON form (`{"name": ..., "arguments": {...}}`).
    static func parseToolCallTags(_ text: String) -> AgentResponse? {
        guard text.contains("<tool_call>") || text.contains("<function=") else { return nil }
        var calls: [AgentCall] = []
        var blocks = text.components(separatedBy: "<tool_call>").dropFirst().map {
            $0.components(separatedBy: "</tool_call>").first ?? $0
        }
        if blocks.isEmpty { blocks = [text] }
        for block in blocks {
            if let function = firstCapture(#"<function=([^>\s]+)\s*>"#, in: block) {
                var call = AgentCall(type: function)
                var other: String?
                for pair in captures(#"<parameter=([^>\s]+)\s*>([\s\S]*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)"#, in: block) {
                    let value = pair[1].trimmingCharacters(in: .whitespacesAndNewlines)
                    switch pair[0].lowercased() {
                    case "query", "q", "keyword", "keywords", "expression", "expr", "formula", "input": call.query = value
                    case "url", "link": call.url = value
                    case "content", "memory", "fact", "preference": call.content = value
                    default: if !value.isEmpty { other = other ?? value }
                    }
                }
                // Unexpected parameter names (e.g. "search_query", "text").
                if call.query == nil, call.url == nil, call.content == nil, let other {
                    switch call.tool {
                    case "browser_url": call.url = other
                    case "save_memory": call.content = other
                    default: call.query = other
                    }
                }
                if call.tool == "save_memory", call.content == nil { call.content = call.query }
                calls.append(call)
            } else if let object = firstJSONObject(in: block), let parsed = parseCandidate(object) {
                calls += parsed.calls
            }
        }
        calls = calls.filter { toolTypes.contains($0.tool) || toolTypes.contains($0.type.lowercased()) }
        return calls.isEmpty ? nil : AgentResponse(reply: "", calls: calls)
    }

    private static func firstCapture(_ pattern: String, in text: String) -> String? {
        captures(pattern, in: text).first?.first
    }

    private static func captures(_ pattern: String, in text: String) -> [[String]] {
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return [] }
        return regex.matches(in: text, range: NSRange(text.startIndex..., in: text)).map { match in
            (1..<match.numberOfRanges).map { Range(match.range(at: $0), in: text).map { String(text[$0]) } ?? "" }
        }
    }

    // MARK: Candidates

    private static func candidates(_ text: String) -> [String] {
        var result = [text]
        if let regex = try? NSRegularExpression(pattern: "```(?:json)?\\s*([\\s\\S]*?)```", options: [.caseInsensitive]) {
            let range = NSRange(text.startIndex..., in: text)
            for match in regex.matches(in: text, range: range) {
                if let r = Range(match.range(at: 1), in: text) { result.append(String(text[r]).trimmingCharacters(in: .whitespacesAndNewlines)) }
            }
        }
        if let object = firstJSONObject(in: text) { result.append(object) }
        var seen = Set<String>()
        return result.filter { !$0.isEmpty && seen.insert($0).inserted }
    }

    static func firstJSONObject(in text: String) -> String? {
        guard let start = text.firstIndex(of: "{") else { return nil }
        var depth = 0, inString = false, escaping = false
        var index = start
        while index < text.endIndex {
            let ch = text[index]
            if escaping { escaping = false }
            else if ch == "\\" && inString { escaping = true }
            else if ch == "\"" { inString.toggle() }
            else if !inString {
                if ch == "{" { depth += 1 }
                if ch == "}" {
                    depth -= 1
                    if depth == 0 { return String(text[start...index]) }
                }
            }
            index = text.index(after: index)
        }
        return nil
    }

    // MARK: JSON

    private static func parseCandidate(_ candidate: String) -> AgentResponse? {
        guard let data = candidate.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed]) else { return nil }
        if let array = json as? [Any] {
            let calls = calls(fromArray: array)
            return calls.isEmpty ? nil : AgentResponse(reply: "", calls: calls)
        }
        guard let object = json as? [String: Any] else { return nil }
        if let call = singleTool(object) { return AgentResponse(reply: "", calls: [call]) }
        return AgentResponse(reply: string(object["reply"]) ?? string(object["answer"]),
                             calls: systemCalls(object["system_calls"] ?? object["tool_calls"]),
                             memoryUpdates: memories(object["memory_updates"]),
                             skillUpdates: skills(object["skill_updates"]))
    }

    private static func parseNested(_ reply: String) -> AgentResponse? {
        for candidate in candidates(reply) where candidate.contains("{") || candidate.contains("[") {
            if let nested = parseCandidate(candidate), !nested.isEmpty { return nested }
        }
        return nil
    }

    private static func singleTool(_ object: [String: Any]) -> AgentCall? {
        let wrapperKeys = ["system_calls", "tool_calls", "reply", "answer", "memory_updates", "skill_updates"]
        guard !wrapperKeys.contains(where: { object[$0] != nil }) else { return nil }
        let explicit = string(object["type"] ?? object["name"] ?? object["tool"])
        var type = explicit.flatMap { toolTypes.contains($0.lowercased()) ? $0 : nil }
        if type == nil {
            if object["url"] != nil { type = "browser_url" } else if object["query"] != nil { type = "web_search" }
        }
        guard let type else { return nil }
        let arguments = object["arguments"] as? [String: Any] ?? object["parameters"] as? [String: Any] ?? object
        return AgentCall(type: type, query: string(arguments["query"]) ?? string(arguments["q"])
                            ?? string(arguments["expression"]) ?? string(arguments["expr"]),
                         url: string(arguments["url"]),
                         content: string(arguments["content"]) ?? string(arguments["memory"]))
    }

    private static func systemCalls(_ value: Any?) -> [AgentCall] {
        switch value {
        case let array as [Any]: calls(fromArray: array)
        case let object as [String: Any]: singleTool(object).map { [$0] } ?? []
        case let text as String: parse(text)?.calls ?? []
        default: []
        }
    }

    private static func calls(fromArray array: [Any]) -> [AgentCall] {
        array.flatMap { element -> [AgentCall] in
            switch element {
            case let object as [String: Any]: singleTool(object).map { [$0] } ?? []
            case let nested as [Any]: calls(fromArray: nested)
            case let text as String: parse(text)?.calls ?? []
            default: []
            }
        }
    }

    private static func memories(_ value: Any?) -> [(content: String, category: String)] {
        guard let array = value as? [Any] else { return [] }
        return array.compactMap { item in
            if let text = item as? String, !text.isEmpty { return (text, "general") }
            if let object = item as? [String: Any], let content = string(object["content"]) ?? string(object["text"]) {
                return (content, string(object["category"]) ?? "general")
            }
            return nil
        }
    }

    private static func skills(_ value: Any?) -> [AgentSkill] {
        guard let array = value as? [Any] else { return [] }
        return array.compactMap { item in
            guard let object = item as? [String: Any],
                  let name = string(object["name"]),
                  let action = string(object["action_template"]) ?? string(object["description"]) else { return nil }
            var triggers: [String] = []
            if let list = object["trigger_keywords"] as? [Any] { triggers = list.compactMap { $0 as? String } }
            else if let text = string(object["trigger_keywords"]) {
                triggers = text.split(whereSeparator: { $0 == "," || $0 == ";" || $0 == "，" }).map { $0.trimmingCharacters(in: .whitespaces) }
            }
            return AgentSkill(name: name, triggers: triggers, actionTemplate: action)
        }
    }

    private static func string(_ value: Any?) -> String? {
        guard let text = value as? String else { return nil }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }
}
