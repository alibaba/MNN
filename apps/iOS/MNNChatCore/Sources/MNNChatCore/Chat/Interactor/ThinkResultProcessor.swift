//
//  ThinkResultProcessor.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/2/11.
//

import Foundation

/// Splits a thinking-model's output stream into the thinking scratchpad and
/// the user-visible answer, fed by native streaming deltas.
///
/// Two wire formats exist in the wild:
/// - the stream literally opens with `<think>` (DeepSeek-style templates), or
/// - the chat template pre-opens the block in the prompt (Qwen3.5 jinja), so
///   the stream only contains the closing `</think>`.
/// `expectThinking` (config-driven) selects the starting mode; a literal
/// opening tag always forces thinking mode regardless.
///
/// Work per delta is proportional to the delta (plus copying the outputs):
/// the closing-tag search resumes where it stopped and edge trimming only
/// inspects the ends, so a long answer no longer gets slower as it grows.
final class ThinkResultProcessor {
    private static let openTag = Array("<think>".utf8)
    private static let closeTag = Array("</think>".utf8)

    private var raw = ""
    /// Thinking content so far, tags stripped.
    private(set) var thinking = ""
    /// Answer content so far (empty until `</think>` arrives in thinking mode).
    private(set) var answer = ""

    /// When true the round starts in thinking mode even without an opening tag.
    var expectThinking: Bool = false

    // Incremental search state, as UTF-8 offsets into `raw`.
    private var searchBodyStart = -1
    private var searchedUpTo = 0
    private var closeStart: Int?
    private var finishedThinking: String?

    func startNewChat() {
        raw = ""
        thinking = ""
        answer = ""
        resetSearch()
    }

    /// Kept for API compatibility with the previous implementation.
    func startGeneration() {}

    /// Feed one native streaming delta.
    func process(delta: String) {
        raw += delta
        recompute()
    }

    private func resetSearch() {
        searchBodyStart = -1
        searchedUpTo = 0
        closeStart = nil
        finishedThinking = nil
    }

    private func recompute() {
        let utf8 = raw.utf8
        let opened = utf8.starts(with: Self.openTag)
        guard opened || expectThinking else {
            // No opening marker and thinking not expected: plain answer.
            thinking = ""
            answer = raw
            return
        }

        let bodyStart = opened ? Self.openTag.count : 0
        if searchBodyStart != bodyStart {
            resetSearch()
            searchBodyStart = bodyStart
            searchedUpTo = bodyStart
        }
        let count = utf8.count
        if closeStart == nil {
            // A tag may straddle the previous search end.
            closeStart = findCloseTag(from: max(bodyStart, searchedUpTo - (Self.closeTag.count - 1)))
            searchedUpTo = count
        }

        if let close = closeStart {
            if finishedThinking == nil { finishedThinking = trimmed(bodyStart, close) }
            thinking = finishedThinking ?? ""
            answer = trimmed(close + Self.closeTag.count, count)
        } else {
            // A closing tag can straddle two native deltas ("...</thi" + "nk>...");
            // hide the partial tail until the rest arrives.
            thinking = trimmed(bodyStart, count - partialCloseTagLength(bodyStart: bodyStart))
            answer = ""
        }
    }

    private func findCloseTag(from start: Int) -> Int? {
        let utf8 = raw.utf8
        let tag = Self.closeTag
        var index = utf8.index(utf8.startIndex, offsetBy: start)
        while let candidate = utf8[index...].firstIndex(of: tag[0]) {
            if utf8[candidate...].starts(with: tag) { return utf8.distance(from: utf8.startIndex, to: candidate) }
            index = utf8.index(after: candidate)
        }
        return nil
    }

    /// Length of the shortest closing-tag prefix the body ends with, or 0.
    private func partialCloseTagLength(bodyStart: Int) -> Int {
        let utf8 = raw.utf8
        let bodyLength = utf8.count - bodyStart
        for length in 1..<Self.closeTag.count where length <= bodyLength {
            if utf8.suffix(length).elementsEqual(Self.closeTag[0..<length]) { return length }
        }
        return 0
    }

    /// `raw[start..<end]` without leading/trailing whitespace and newlines
    /// (the template emits `<think>\n` and `</think>\n\n`). Only the edges
    /// are inspected; offsets always fall right after ASCII tag bytes.
    private func trimmed(_ start: Int, _ end: Int) -> String {
        guard end > start else { return "" }
        let utf8 = raw.utf8
        let scalars = raw.unicodeScalars
        var lower = utf8.index(utf8.startIndex, offsetBy: start)
        var upper = utf8.index(utf8.startIndex, offsetBy: end)
        let whitespace = CharacterSet.whitespacesAndNewlines
        while lower < upper, whitespace.contains(scalars[lower]) {
            lower = scalars.index(after: lower)
        }
        while lower < upper {
            let previous = scalars.index(before: upper)
            guard whitespace.contains(scalars[previous]) else { break }
            upper = previous
        }
        return String(decoding: utf8[lower..<upper], as: UTF8.self)
    }
}
