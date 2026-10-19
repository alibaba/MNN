//
//  PreTokenizer.swift
//
//
//  Created by Pedro Cuenca on 18/7/23.
//

import Foundation
import Hub

public enum PreTokenizerOption: String {
    case firstSection
}

public typealias PreTokenizerOptions = Set<PreTokenizerOption>

public protocol PreTokenizer {
    func preTokenize(text: String, options: PreTokenizerOptions) -> [String]
    func preTokenize(texts: [String], options: PreTokenizerOptions) -> [String]
    func callAsFunction(texts: [String], options: PreTokenizerOptions) -> [String]
    func callAsFunction(text: String, options: PreTokenizerOptions) -> [String]

    init(config: Config)
}

extension PreTokenizer {
    func preTokenize(texts: [String], options: PreTokenizerOptions = [.firstSection]) -> [String] {
        texts.flatMap { preTokenize(text: $0, options: options) }
    }

    func callAsFunction(texts: [String], options: PreTokenizerOptions = [.firstSection]) -> [String] {
        preTokenize(texts: texts, options: options)
    }

    func callAsFunction(text: String, options: PreTokenizerOptions = [.firstSection]) -> [String] {
        preTokenize(text: text, options: options)
    }
}

enum PreTokenizerType: String {
    case Sequence
    case ByteLevel
    case Punctuation
    case Digits
    case Split
    case Whitespace
    case WhitespaceSplit
    case Metaspace
    case BertPreTokenizer
    /// Several more to be supported
    case Unknown = ""
}

struct PreTokenizerFactory {
    static func fromConfig(config: Config?) -> PreTokenizer? {
        guard let config else { return nil }
        guard let typeName = config.type.string() else { return nil }
        let type = PreTokenizerType(rawValue: typeName)
        switch type {
        case .Sequence: return PreTokenizerSequence(config: config)
        case .ByteLevel: return ByteLevelPreTokenizer(config: config)
        case .Punctuation: return PunctuationPreTokenizer(config: config)
        case .Digits: return DigitsPreTokenizer(config: config)
        case .Split: return SplitPreTokenizer(config: config)
        case .Whitespace, .WhitespaceSplit: return WhitespacePreTokenizer(config: config)
        case .Metaspace: return MetaspacePreTokenizer(config: config)
        case .BertPreTokenizer: return BertPreTokenizer(config: config)
        default: fatalError("Unsupported PreTokenizer type: \(typeName)")
        }
    }
}

class BertPreTokenizer: PreTokenizer {
    let re: String

    required init(config: Config) {
        // Ref: https://github.com/huggingface/transformers.js/blob/27920d84831e323275b38f0b5186644b7936e1a2/src/tokenizers.js#L1002
        re = "[^\\s\(Constants.PUNCTUATION_REGEX)]+|[\(Constants.PUNCTUATION_REGEX)]"
    }

    func preTokenize(text: String, options: PreTokenizerOptions = [.firstSection]) -> [String] {
        text.ranges(of: re).map { String(text[$0]) }
    }
}

class PreTokenizerSequence: PreTokenizer {
    let preTokenizers: [PreTokenizer]

    required init(config: Config) {
        guard let configs = config.pretokenizers.array() else { fatalError("No pretokenizers in Sequence") }
        preTokenizers = configs.compactMap { PreTokenizerFactory.fromConfig(config: $0) }
    }

    func preTokenize(text: String, options: PreTokenizerOptions = [.firstSection]) -> [String] {
        preTokenizers.reduce([text]) { current, preTokenizer in
            preTokenizer(texts: current, options: options)
        }
    }
}

class WhitespacePreTokenizer: PreTokenizer {
    let re: String

    required init(config: Config) {
        re = #"\S+"#
    }

    func preTokenize(text: String, options: PreTokenizerOptions = [.firstSection]) -> [String] {
        text.ranges(of: re).map { String(text[$0]) }
    }
}

/// PreTokenizer that replaces spaces with the given replacement character, adds a prefix space if requested,
class MetaspacePreTokenizer: PreTokenizer {
    /// Whether to add a prefix space to the first token
    let addPrefixSpace: Bool

    /// Replacement character
    let replacement: String

    /// Optional string representation of the replacement character.
    let stringReplacement: String

    enum PrependScheme: String {
        case first
        case never
        case always

        static var defaultScheme: PrependScheme { .always }
        static func from(rawValue value: String?) -> PrependScheme {
            guard let value else { return defaultScheme }
            return PrependScheme(rawValue: value) ?? defaultScheme
        }
    }

    /// The metaspace prepend scheme, see https://github.com/huggingface/tokenizers/pull/1357
    let prependScheme: PrependScheme

    required init(config: Config) {
        addPrefixSpace = config.addPrefixSpace.boolean(or: false)
        replacement = config.replacement.string(or: " ")
        stringReplacement = config.strRep.string(or: replacement)
        prependScheme = PrependScheme.from(rawValue: config.prependScheme.string())
    }

    /// https://github.com/huggingface/tokenizers/blob/accd0650b802f2180df40ef1def3bce32156688e/tokenizers/src/pre_tokenizers/metaspace.rs#L114
    /// https://github.com/xenova/transformers.js/blob/b07336d8f7ff57453cc164cc68aead2a79cbd57e/src/tokenizers.js#L2153
    func preTokenize(text: String, options: PreTokenizerOptions = [.firstSection]) -> [String] {
        let normalized = text.replacingOccurrences(of: " ", with: stringReplacement)

        // We add a prefix space if:
        //  (1) The addPrefixSpace option is enabled and the normalized
        //      token does not already start with the replacement character.
        //  and (2) either:
        //  (a) prependScheme is 'always'
        //  (b) prependScheme is 'first' and this is the first section
        // FIXME: (2b) always prepends, we are not passing section info

        var prepend = ""
        if addPrefixSpace, !normalized.hasPrefix(replacement) {
            if prependScheme == .always {
                prepend = stringReplacement
            }
            if prependScheme == .first, options.contains(.firstSection) {
                prepend = stringReplacement
            }
        }

        // Split in `MergedWithNext` mode, although usually the input to this function is already pre-tokenized
        // https://github.com/huggingface/tokenizers/blob/accd0650b802f2180df40ef1def3bce32156688e/tokenizers/src/pre_tokenizers/metaspace.rs#L127
        return (prepend + normalized).split(by: replacement, behavior: .mergedWithNext)
    }
}

class ByteLevelPreTokenizer: PreTokenizer {
    let addPrefixSpace: Bool
    let trimOffsets: Bool
    let useRegex: Bool
    let RE = #"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#

    required init(config: Config) {
        addPrefixSpace = config.addPrefixSpace.boolean(or: false)
        trimOffsets = config.trimOffsets.boolean(or: true)
        useRegex = config.useRegex.boolean(or: true)
    }

    func preTokenize(text: String, options: PreTokenizerOptions = [.firstSection]) -> [String] {
        // Split on whitespace and punctuation
        let tokens = useRegex ? text.ranges(of: RE).map { String(text[$0]) } : [text]
        return tokens.map { token in
            if addPrefixSpace, !token.hasPrefix(" ") {
                return " " + token
            }
            return token
        }.map { token in
            Array(token.utf8).map { byteEncoder[$0]! }.joined()
        }
    }
}

class PunctuationPreTokenizer: PreTokenizer {
    let re: String

    required init(config: Config) {
        re = "[^\(Constants.PUNCTUATION_REGEX)]+|[\(Constants.PUNCTUATION_REGEX)]+"
    }

    func preTokenize(text: String, options: PreTokenizerOptions = [.firstSection]) -> [String] {
        // Ref: https://github.com/xenova/transformers.js/blob/27920d84831e323275b38f0b5186644b7936e1a2/src/tokenizers.js#L1138
        text.ranges(of: re).map { String(text[$0]) }
    }
}

class DigitsPreTokenizer: PreTokenizer {
    let re: String

    required init(config: Config) {
        let individualDigits = config.individualDigits.boolean(or: false)
        re = "[^\\d]+|\\d\(individualDigits ? "" : "+")"
    }

    func preTokenize(text: String, options: PreTokenizerOptions = [.firstSection]) -> [String] {
        text.ranges(of: re).map { String(text[$0]) }
    }
}

class SplitPreTokenizer: PreTokenizer {
    let pattern: StringSplitPattern?
    let invert: Bool

    required init(config: Config) {
        pattern = StringSplitPattern.from(config: config)
        invert = config.invert.boolean(or: false)
    }

    func preTokenize(text: String, options: PreTokenizerOptions = [.firstSection]) -> [String] {
        guard let pattern else { return [text] }
        return pattern.split(text, invert: invert)
    }
}

enum StringSplitPattern {
    case regexp(regexp: String)
    case string(pattern: String)
}

extension StringSplitPattern {
    func split(_ text: String, invert: Bool = true) -> [String] {
        switch self {
        case let .regexp(regexp):
            text.split(by: regexp, includeSeparators: true)
        case let .string(substring):
            text.split(by: substring, options: [], includeSeparators: !invert)
        }
    }
}

extension StringSplitPattern {
    static func from(config: Config) -> StringSplitPattern? {
        if let pattern = config.pattern.String.string() {
            return StringSplitPattern.string(pattern: pattern)
        }
        if let pattern = config.pattern.Regex.string() {
            return StringSplitPattern.regexp(regexp: pattern)
        }
        return nil
    }
}

public extension String {
    func ranges(of string: String, options: CompareOptions = .regularExpression) -> [Range<Index>] {
        var result: [Range<Index>] = []
        var start = startIndex
        while let range = range(of: string, options: options, range: start..<endIndex) {
            result.append(range)
            start = range.lowerBound < range.upperBound ? range.upperBound : index(range.lowerBound, offsetBy: 1, limitedBy: endIndex) ?? endIndex
        }
        return result
    }

    func split(by string: String, options: CompareOptions = .regularExpression, includeSeparators: Bool = false, omittingEmptySubsequences: Bool = true) -> [String] {
        var result: [String] = []
        var start = startIndex
        while let range = range(of: string, options: options, range: start..<endIndex) {
            // Prevent empty strings
            if omittingEmptySubsequences, start < range.lowerBound {
                result.append(String(self[start..<range.lowerBound]))
            }
            if includeSeparators {
                result.append(String(self[range]))
            }
            start = range.upperBound
        }

        if omittingEmptySubsequences, start < endIndex {
            result.append(String(self[start...]))
        }
        return result
    }

    /// This version supports capture groups, wheres the one above doesn't
    func split(by captureRegex: NSRegularExpression) -> [String] {
        // Find the matching capture groups
        let selfRange = NSRange(startIndex..<endIndex, in: self)
        let matches = captureRegex.matches(in: self, options: [], range: selfRange)

        if matches.isEmpty { return [self] }

        var result: [String] = []
        var start = startIndex
        for match in matches {
            // Safely move the prefix end to the start of the current match
            let safePrefixEnd = index(startIndex, offsetBy: match.range.lowerBound, limitedBy: endIndex) ?? endIndex
            if start < safePrefixEnd {
                result.append(String(self[start..<safePrefixEnd]))
            }

            // Safely move the start index to the end of the current match
            let matchEndIndex = index(startIndex, offsetBy: match.range.upperBound, limitedBy: endIndex) ?? endIndex
            start = matchEndIndex

            // Append separator, supporting capture groups
            for r in (0..<match.numberOfRanges).reversed() {
                let matchRange = match.range(at: r)
                if let sepRange = Range(matchRange, in: self) {
                    result.append(String(self[sepRange]))
                    break
                }
            }
        }

        // Append remaining suffix
        if start < endIndex {
            result.append(String(self[start...]))
        }

        return result
    }
}

public enum SplitDelimiterBehavior {
    case removed
    case isolated
    case mergedWithPrevious
    case mergedWithNext
}

public extension String {
    func split(by string: String, options: CompareOptions = .regularExpression, behavior: SplitDelimiterBehavior) -> [String] {
        func mergedWithNext(ranges: [Range<String.Index>]) -> [Range<String.Index>] {
            var merged: [Range<String.Index>] = []
            var currentStart = startIndex
            for range in ranges {
                if range.lowerBound == startIndex { continue }
                let mergedRange = currentStart..<range.lowerBound
                currentStart = range.lowerBound
                merged.append(mergedRange)
            }
            if currentStart < endIndex {
                merged.append(currentStart..<endIndex)
            }
            return merged
        }

        func mergedWithPrevious(ranges: [Range<String.Index>]) -> [Range<String.Index>] {
            var merged: [Range<String.Index>] = []
            var currentStart = startIndex
            for range in ranges {
                let mergedRange = currentStart..<range.upperBound
                currentStart = range.upperBound
                merged.append(mergedRange)
            }
            if currentStart < endIndex {
                merged.append(currentStart..<endIndex)
            }
            return merged
        }

        switch behavior {
        case .removed:
            return split(by: string, options: options, includeSeparators: false)
        case .isolated:
            return split(by: string, options: options, includeSeparators: true)
        case .mergedWithNext:
            // Obtain ranges and merge them
            // "the-final--countdown" -> (3, 4), (9, 10), (10, 11) -> (start, 2), (3, 8), (9, 9), (10, end)
            let ranges = ranges(of: string, options: options)
            let merged = mergedWithNext(ranges: ranges)
            return merged.map { String(self[$0]) }
        case .mergedWithPrevious:
            // Obtain ranges and merge them
            // "the-final--countdown" -> (3, 4), (9, 10), (10, 11) -> (start, 3), (4, 9), (10, 10), (11, end)
            let ranges = ranges(of: string, options: options)
            let merged = mergedWithPrevious(ranges: ranges)
            return merged.map { String(self[$0]) }
        }
    }
}
