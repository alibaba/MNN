import Foundation

/// Executes agent tool calls. Network tools fetch pages over plain HTTPS; the
/// model itself never leaves the device.
enum AgentTools {
    struct SearchHit: Equatable {
        let text: String
        let url: String
    }

    static let maxSearchResults = 5
    /// Page text handed to the model. Small models have small contexts.
    static let maxPageCharacters = 3000

    private static let userAgent = "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 "
        + "(KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1"

    /// Engines are tried in order. Bing answers scripted requests from China
    /// with unrelated results, so domestic engines come first.
    private static let engines: [(name: String, url: (String) -> String)] = [
        ("Sogou", { "https://m.sogou.com/web/searchList.jsp?keyword=\($0)" }),
        ("360", { "https://m.so.com/s?q=\($0)" }),
        ("Bing", { "https://www.bing.com/search?q=\($0)" }),
    ]

    private static let session: URLSession = {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 10
        configuration.timeoutIntervalForResource = 15
        configuration.httpAdditionalHeaders = [
            "User-Agent": userAgent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        ]
        return URLSession(configuration: configuration)
    }()

    /// Runs one call and returns the observation text for the model.
    static func execute(_ call: AgentCall) async -> String {
        do {
            switch call.tool {
            case "get_current_time": return currentTime()
            case "calculator": return calculate(call.query ?? call.content ?? "")
            case "web_search": return try await webSearch(call.query ?? call.url ?? "")
            case "browser_url": return try await readPage(call.url ?? call.query ?? "")
            default: return "[TOOL_ERROR] Unknown tool \"\(call.type)\". Available: get_current_time, web_search, browser_url, calculator, save_memory."
            }
        } catch is CancellationError {
            return "[TOOL_ERROR] Cancelled."
        } catch {
            return "[TOOL_ERROR] \(call.tool) failed: \(error.localizedDescription)"
        }
    }

    // MARK: Time

    static func currentTime(now: Date = Date()) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyy-MM-dd HH:mm:ss"
        let weekday = DateFormatter()
        weekday.locale = Locale(identifier: "en_US_POSIX")
        weekday.dateFormat = "EEEE"
        return """
        [TIME_RESULT]
        datetime: \(formatter.string(from: now))
        weekday: \(weekday.string(from: now))
        timezone: \(TimeZone.current.identifier)
        """
    }

    // MARK: Search

    static func webSearch(_ query: String) async throws -> String {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "[SEARCH_ERROR] Empty query." }
        let encoded = trimmed.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed.subtracting(CharacterSet(charactersIn: "&+=?#"))) ?? trimmed
        var lastError = "No search results were returned."
        for engine in engines {
            try Task.checkCancellation()
            guard let url = URL(string: engine.url(encoded)) else { continue }
            do {
                let html = try await fetch(url)
                let hits = searchHits(in: html, base: url)
                if hits.count >= 2 {
                    var lines = ["[SEARCH_RESULT]", "query: \(trimmed)", "source: \(engine.name)"]
                    for (index, hit) in hits.prefix(maxSearchResults).enumerated() {
                        lines.append("\(index + 1). \(hit.text)")
                        lines.append("   url: \(hit.url)")
                    }
                    return lines.joined(separator: "\n")
                }
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                lastError = error.localizedDescription
            }
        }
        return "[SEARCH_ERROR] \(lastError)"
    }

    /// Engine-agnostic result extraction: anchors that lead off the search
    /// site (directly or through a `url=` redirect parameter), merged by
    /// destination so a result's title and snippet anchors become one entry.
    static func searchHits(in html: String, base: URL) -> [SearchHit] {
        let page = removeBlocks(html, tags: ["script", "style", "noscript"])
        let engineDomain = base.host.map(registrableName) ?? ""
        var order: [String] = []
        var texts: [String: [String]] = [:]
        for anchor in matches(of: #"<a\b([^>]*)>([\s\S]*?)</a>"#, in: page) {
            guard anchor.count == 3, let href = attribute("href", in: anchor[1]) else { continue }
            let text = plainText(anchor[2])
            guard text.count >= 8, let destination = destinationURL(href: decodeEntities(href), attributes: anchor[1], base: base),
                  let host = URL(string: destination)?.host?.lowercased(), engineDomain.isEmpty || !host.contains(engineDomain)
            else { continue }
            if texts[destination] == nil { order.append(destination); texts[destination] = [] }
            if !(texts[destination]?.contains(text) ?? true) { texts[destination]?.append(text) }
        }
        return order.compactMap { url in
            guard let parts = texts[url], !parts.isEmpty else { return nil }
            return SearchHit(text: String(parts.joined(separator: " — ").prefix(240)), url: url)
        }
    }

    private static func destinationURL(href: String, attributes: String, base: URL) -> String? {
        for name in ["mu", "data-url", "data-mdurl"] {
            if let value = attribute(name, in: attributes), value.hasPrefix("http") { return value }
        }
        if let components = URLComponents(string: href) {
            for item in components.queryItems ?? [] where ["url", "u", "target"].contains(item.name) {
                if let value = item.value, value.hasPrefix("http") { return value }
            }
        }
        guard let resolved = URL(string: href, relativeTo: base)?.absoluteURL,
              ["http", "https"].contains(resolved.scheme ?? "") else { return nil }
        return resolved.absoluteString
    }

    /// "m.sogou.com" -> "sogou", "www.bing.com" -> "bing".
    private static func registrableName(_ host: String) -> String {
        let parts = host.split(separator: ".")
        return parts.count >= 2 ? String(parts[parts.count - 2]) : host
    }

    // MARK: Page reading

    static func readPage(_ raw: String) async throws -> String {
        var text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if !text.lowercased().hasPrefix("http://") && !text.lowercased().hasPrefix("https://") { text = "https://" + text }
        guard let url = URL(string: text), let host = url.host, !host.isEmpty else {
            return "[BROWSE_ERROR] Invalid URL; only http/https pages are supported."
        }
        let html = try await fetch(url)
        let title = matches(of: #"<title[^>]*>([\s\S]*?)</title>"#, in: html).first.map { plainText($0[1]) } ?? ""
        let body = readableText(html)
        guard !body.isEmpty else { return "[BROWSE_ERROR] Page content could not be read." }
        return "[BROWSE_RESULT]\nurl: \(url.absoluteString)\n" + (title.isEmpty ? "" : "title: \(title)\n")
            + "text:\n" + String(body.prefix(maxPageCharacters))
    }

    static func readableText(_ html: String) -> String {
        let body = matches(of: #"<body[^>]*>([\s\S]*?)</body>"#, in: html).first?[1] ?? html
        var cleaned = removeBlocks(body, tags: ["script", "style", "noscript", "svg", "nav", "footer", "header"])
        cleaned = replace(#"<br\s*/?>"#, in: cleaned, with: "\n")
        cleaned = replace(#"</(p|div|li|tr|h[1-6]|section|article|table)>"#, in: cleaned, with: "\n")
        return plainText(cleaned, keepNewlines: true)
    }

    private static func fetch(_ url: URL) async throws -> String {
        let (data, response) = try await session.data(from: url)
        if let http = response as? HTTPURLResponse, http.statusCode >= 400 {
            throw URLError(.badServerResponse, userInfo: [NSLocalizedDescriptionKey: "HTTP \(http.statusCode)"])
        }
        var encoding = String.Encoding.utf8
        if let name = response.textEncodingName {
            let cf = CFStringConvertIANACharSetNameToEncoding(name as CFString)
            if cf != kCFStringEncodingInvalidId { encoding = String.Encoding(rawValue: CFStringConvertEncodingToNSStringEncoding(cf)) }
        }
        return String(data: data, encoding: encoding) ?? String(decoding: data, as: UTF8.self)
    }

    // MARK: HTML helpers

    static func plainText(_ html: String, keepNewlines: Bool = false) -> String {
        var text = replace("<[^>]+>", in: html, with: " ")
        text = decodeEntities(text)
        if keepNewlines {
            text = replace(#"[ \t\r\f\u00a0]+"#, in: text, with: " ")
            text = replace(#"\n\s*\n+"#, in: text, with: "\n")
            text = replace(#" *\n *"#, in: text, with: "\n")
        } else {
            text = replace(#"\s+"#, in: text, with: " ")
        }
        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func decodeEntities(_ text: String) -> String {
        var result = text
        for (entity, value) in [("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&quot;", "\""), ("&#39;", "'"),
                                ("&#x27;", "'"), ("&nbsp;", " "), ("&ensp;", " "), ("&emsp;", " ")] {
            result = result.replacingOccurrences(of: entity, with: value)
        }
        return result
    }

    private static func removeBlocks(_ html: String, tags: [String]) -> String {
        tags.reduce(html) { replace("<\($1)\\b[\\s\\S]*?</\($1)>", in: $0, with: " ") }
    }

    private static func attribute(_ name: String, in attributes: String) -> String? {
        let pattern = "\\b\(name)\\s*=\\s*(?:\"([^\"]*)\"|'([^']*)')"
        guard let match = matches(of: pattern, in: attributes).first else { return nil }
        let value = match.count > 1 && !match[1].isEmpty ? match[1] : (match.count > 2 ? match[2] : "")
        return value.isEmpty ? nil : value
    }

    private static func matches(of pattern: String, in text: String) -> [[String]] {
        guard let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]) else { return [] }
        return regex.matches(in: text, range: NSRange(text.startIndex..., in: text)).map { match in
            (0..<match.numberOfRanges).map { index in
                Range(match.range(at: index), in: text).map { String(text[$0]) } ?? ""
            }
        }
    }

    private static func replace(_ pattern: String, in text: String, with template: String) -> String {
        guard let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]) else { return text }
        return regex.stringByReplacingMatches(in: text, range: NSRange(text.startIndex..., in: text), withTemplate: template)
    }
}

// MARK: - Search query refinement

extension AgentTools {
    private static let fillerPhrases = ["请帮我", "帮我", "帮忙", "麻烦", "请你", "请", "联网", "上网", "网上", "搜索一下", "搜一下",
                                        "搜索", "查一下", "查一查", "查查", "查询", "看一下", "看下", "告诉我", "一下", "能不能", "可以"]
    private static let englishFillers: Set<String> = ["what", "is", "it", "the", "a", "an", "of", "for", "to", "in", "and",
                                                      "does", "do", "how", "about", "are", "which", "who", "me", "please"]

    /// Small models often reduce a Chinese question to a few English words
    /// ("MNN what is it"), which search engines answer with unrelated pages.
    /// Keep the user's own wording in the query: a non-Chinese query for a
    /// Chinese question is rebuilt from the question plus the model's
    /// meaningful terms; a Chinese query gets the question's missing names.
    static func refinedQuery(_ query: String, request: String) -> String {
        let model = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard containsCJK(request) else { return model }
        let core = coreText(of: request)
        guard !core.isEmpty else { return model }
        if !containsCJK(model) {
            let extra = model.split(whereSeparator: { $0.isWhitespace })
                .map(String.init)
                .filter { !englishFillers.contains($0.lowercased()) && !core.lowercased().contains($0.lowercased()) }
                .prefix(4)
            return ([core] + extra).joined(separator: " ")
        }
        // Names and Latin terms from the question that the model dropped.
        let missing = core.split(whereSeparator: { $0.isWhitespace })
            .map(String.init)
            .filter { $0.count >= 2 && !containsCJK($0) && !model.lowercased().contains($0.lowercased()) }
            .prefix(3)
        return ([model] + missing).joined(separator: " ")
    }

    /// The question without request phrasing and punctuation, e.g.
    /// "MNN 是什么？请联网查一下。" -> "MNN 是什么".
    static func coreText(of request: String) -> String {
        var text = request
        for phrase in fillerPhrases { text = text.replacingOccurrences(of: phrase, with: " ") }
        text = text.unicodeScalars.map { CharacterSet.punctuationCharacters.contains($0) || CharacterSet.symbols.contains($0) ? " " : String($0) }.joined()
        let words = text.split(whereSeparator: { $0.isWhitespace }).map(String.init)
        return String(words.joined(separator: " ").prefix(30)).trimmingCharacters(in: .whitespaces)
    }

    static func containsCJK(_ text: String) -> Bool {
        text.unicodeScalars.contains { (0x4E00...0x9FFF).contains($0.value) || (0x3400...0x4DBF).contains($0.value) }
    }
}

// MARK: - Calculator

extension AgentTools {
    enum CalculatorError: Error { case invalid(String) }

    /// Evaluates an arithmetic expression for the `calculator` tool.
    /// Supports + - * / % ^, parentheses, unary minus, sqrt/abs/round/floor/
    /// ceil/ln/log, pi and e. A hand-written parser: NSExpression raises an
    /// uncatchable exception on malformed input, and models write plenty.
    static func calculate(_ expression: String) -> String {
        let normalized = normalizeExpression(expression)
        guard !normalized.isEmpty else { return "[CALC_ERROR] Empty expression." }
        do {
            var parser = ArithmeticParser(normalized)
            let value = try parser.parse()
            guard value.isFinite else { return "[CALC_ERROR] The result is not a finite number (division by zero?)." }
            return "[CALC_RESULT]\nexpression: \(normalized)\nresult: \(formatNumber(value))"
        } catch CalculatorError.invalid(let reason) {
            return "[CALC_ERROR] \(reason) Expression: \(normalized). Write plain arithmetic such as 225.83*7.12."
        } catch {
            return "[CALC_ERROR] Invalid expression: \(normalized)"
        }
    }

    static func normalizeExpression(_ text: String) -> String {
        var s = text
        // LaTeX that models write out of habit.
        for (from, to) in [("\\*", "*"), ("\\times", "*"), ("\\cdot", "*"), ("\\div", "/"), ("\\left", ""), ("\\right", ""),
                           ("\\(", ""), ("\\)", ""), ("$", ""), ("`", "")] {
            s = s.replacingOccurrences(of: from, with: to)
        }
        s = s.replacingOccurrences(of: #"\\frac\{([^{}]*)\}\{([^{}]*)\}"#, with: "(($1)/($2))", options: .regularExpression)
        for (from, to) in [("×", "*"), ("✕", "*"), ("÷", "/"), ("（", "("), ("）", ")"), ("－", "-"), ("＋", "+"),
                           ("**", "^"), ("π", "pi"), ("−", "-"), ("＝", "="), ("　", " ")] {
            s = s.replacingOccurrences(of: from, with: to)
        }
        if let eq = s.firstIndex(of: "=") { s = String(s[..<eq]) }       // "2*3=" or "x = 2*3" both keep the formula side
        // Thousands separators: "1,234.5" -> "1234.5".
        s = s.replacingOccurrences(of: #"(?<=\d),(?=\d{3}\b)"#, with: "", options: .regularExpression)
        return s.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func formatNumber(_ value: Double) -> String {
        if value == value.rounded(), abs(value) < 1e15 { return String(format: "%.0f", value) }
        var text = String(format: "%.10f", value)
        while text.hasSuffix("0") { text.removeLast() }
        if text.hasSuffix(".") { text.removeLast() }
        return text
    }

    private struct ArithmeticParser {
        private let chars: [Character]
        private var index = 0
        private var depth = 0

        init(_ text: String) { chars = Array(text) }

        mutating func parse() throws -> Double {
            let value = try expression()
            skipSpaces()
            guard index == chars.count else { throw CalculatorError.invalid("Unexpected \"\(chars[index])\".") }
            return value
        }

        private mutating func skipSpaces() { while index < chars.count, chars[index].isWhitespace { index += 1 } }

        private mutating func peek() -> Character? { skipSpaces(); return index < chars.count ? chars[index] : nil }

        private mutating func expression() throws -> Double {
            var value = try term()
            while let op = peek(), op == "+" || op == "-" {
                index += 1
                let rhs = try term()
                value = op == "+" ? value + rhs : value - rhs
            }
            return value
        }

        private mutating func term() throws -> Double {
            var value = try power()
            while let op = peek(), op == "*" || op == "/" || op == "%" {
                index += 1
                let rhs = try power()
                switch op {
                case "*": value *= rhs
                case "/": value /= rhs
                default: value = value.truncatingRemainder(dividingBy: rhs)
                }
            }
            return value
        }

        private mutating func power() throws -> Double {
            let base = try unary()
            if peek() == "^" {
                index += 1
                return pow(base, try power())      // right associative
            }
            return base
        }

        private mutating func unary() throws -> Double {
            if peek() == "-" { index += 1; return -(try unary()) }
            if peek() == "+" { index += 1; return try unary() }
            return try primary()
        }

        private mutating func primary() throws -> Double {
            guard let ch = peek() else { throw CalculatorError.invalid("The expression ends too early.") }
            if ch == "(" {
                depth += 1
                guard depth < 64 else { throw CalculatorError.invalid("Too deeply nested.") }
                index += 1
                let value = try expression()
                guard peek() == ")" else { throw CalculatorError.invalid("Missing \")\".") }
                index += 1
                depth -= 1
                return percent(value)
            }
            if ch.isNumber || ch == "." {
                let start = index
                while index < chars.count, chars[index].isNumber || chars[index] == "." { index += 1 }
                if index < chars.count, chars[index] == "e" || chars[index] == "E",
                   index + 1 < chars.count, chars[index + 1].isNumber || chars[index + 1] == "-" || chars[index + 1] == "+" {
                    index += 2
                    while index < chars.count, chars[index].isNumber { index += 1 }
                }
                guard let value = Double(String(chars[start..<index])) else {
                    throw CalculatorError.invalid("Bad number \"\(String(chars[start..<index]))\".")
                }
                return percent(value)
            }
            if ch.isLetter {
                let start = index
                while index < chars.count, chars[index].isLetter { index += 1 }
                let name = String(chars[start..<index]).lowercased()
                switch name {
                case "pi": return .pi
                case "e": return M_E
                default: break
                }
                guard peek() == "(" else { throw CalculatorError.invalid("Unknown name \"\(name)\".") }
                let argument = try primary()
                switch name {
                case "sqrt": return argument.squareRoot()
                case "abs": return abs(argument)
                case "round": return argument.rounded()
                case "floor": return argument.rounded(.down)
                case "ceil": return argument.rounded(.up)
                case "ln": return log(argument)
                case "log": return log10(argument)
                default: throw CalculatorError.invalid("Unknown function \"\(name)\".")
                }
            }
            throw CalculatorError.invalid("Unexpected \"\(ch)\".")
        }

        /// "15%" means 0.15 when the % is not followed by an operand.
        private mutating func percent(_ value: Double) -> Double {
            guard peek() == "%" else { return value }
            var look = index + 1
            while look < chars.count, chars[look].isWhitespace { look += 1 }
            if look < chars.count, chars[look].isNumber || chars[look] == "(" || chars[look] == "." { return value }
            index += 1
            return value / 100
        }
    }
}

// MARK: - Arithmetic check

extension AgentTools {
    /// Small models often compute in their head and get it wrong
    /// ("124.44 × 7.2 = 8959.68"). Recomputes every "expression = result"
    /// written in the answer and replaces a wrong result.
    static func correctArithmetic(in text: String) -> String {
        let number = #"\d[\d,]*(?:\.\d+)?"#
        let operand = #"\(?\s*"# + number + #"\s*\)?%?"#
        let pattern = "(" + operand + #"(?:\s*[×xX*÷/+\-]\s*"# + operand + ")+)" + #"\s*(?:=|≈|约等于|约为|约)\s*\**\s*[$¥￥]?\s*("#
            + number + #")(?!\s*[×xX*÷/+\-]?\s*\d)(?!\s*[×xX*÷/])"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return text }
        var result = text
        let matches = regex.matches(in: text, range: NSRange(text.startIndex..., in: text))
        for match in matches.reversed() {
            guard let exprRange = Range(match.range(at: 1), in: text),
                  let valueRange = Range(match.range(at: 2), in: result),
                  let claimed = Double(result[valueRange].replacingOccurrences(of: ",", with: "")) else { continue }
            let expression = String(text[exprRange])
            // Plain date-like or phone-like runs ("2026-10-17") are not arithmetic.
            if expression.range(of: #"^\d{4}-\d{1,2}-\d{1,2}$"#, options: .regularExpression) != nil { continue }
            let output = calculate(expression)
            guard output.hasPrefix("[CALC_RESULT]"),
                  let line = output.components(separatedBy: "\n").first(where: { $0.hasPrefix("result: ") }),
                  let correct = Double(line.dropFirst(8)) else { continue }
            let tolerance = max(0.011, abs(correct) * 0.002)
            guard abs(claimed - correct) > tolerance else { continue }
            let rounded = (correct * 100).rounded() / 100
            result.replaceSubrange(valueRange, with: formatNumber(rounded))
        }
        return result
    }
}
