//
//  PostProcessor.swift
//
//
//  Created by Pedro Cuenca on 17/7/23.
//

import Foundation
import Hub

public protocol PostProcessor {
    func postProcess(tokens: [String], tokensPair: [String]?, addSpecialTokens: Bool) -> [String]
    func callAsFunction(tokens: [String], tokensPair: [String]?, addSpecialTokens: Bool) -> [String]

    init(config: Config)
}

extension PostProcessor {
    func callAsFunction(tokens: [String], tokensPair: [String]? = nil, addSpecialTokens: Bool = true) -> [String] {
        postProcess(tokens: tokens, tokensPair: tokensPair, addSpecialTokens: addSpecialTokens)
    }
}

enum PostProcessorType: String {
    case TemplateProcessing
    case ByteLevel
    case RobertaProcessing
    case BertProcessing
    case Sequence
}

struct PostProcessorFactory {
    static func fromConfig(config: Config?) -> PostProcessor? {
        guard let config else { return nil }
        guard let typeName = config.type.string() else { return nil }
        let type = PostProcessorType(rawValue: typeName)
        switch type {
        case .TemplateProcessing: return TemplateProcessing(config: config)
        case .ByteLevel: return ByteLevelPostProcessor(config: config)
        case .RobertaProcessing: return RobertaProcessing(config: config)
        case .BertProcessing: return BertProcessing(config: config)
        case .Sequence: return SequenceProcessing(config: config)
        default: fatalError("Unsupported PostProcessor type: \(typeName)")
        }
    }
}

class TemplateProcessing: PostProcessor {
    let single: [Config]
    let pair: [Config]

    public required init(config: Config) {
        guard let single = config.single.array() else { fatalError("Missing `single` processor configuration") }
        guard let pair = config.pair.array() else { fatalError("Missing `pair` processor configuration") }

        self.single = single
        self.pair = pair
    }

    func postProcess(tokens: [String], tokensPair: [String]? = nil, addSpecialTokens: Bool = true) -> [String] {
        let config = tokensPair == nil ? single : pair

        var toReturn: [String] = []
        for item in config {
            if let id = item.SpecialToken.id.string() {
                if addSpecialTokens {
                    toReturn.append(id)
                }
            } else if item.Sequence.id.string() == "A" {
                toReturn += tokens
            } else if item.Sequence.id.string() == "B" {
                toReturn += tokensPair!
            }
        }
        return toReturn
    }
}

class ByteLevelPostProcessor: PostProcessor {
    public required init(config: Config) { }
    func postProcess(tokens: [String], tokensPair: [String]? = nil, addSpecialTokens: Bool = true) -> [String] { tokens }
}

class RobertaProcessing: PostProcessor {
    private let sep: (UInt, String)
    private let cls: (UInt, String)
    /// Trim all remaining space, or leave one space character if `addPrefixSpace` is `true`.
    private let trimOffset: Bool
    /// Keep one space character on each side. Depends on `trimOffsets` being `true`.
    private let addPrefixSpace: Bool

    public required init(config: Config) {
        guard let sep = config.sep.token() else { fatalError("Missing `sep` processor configuration") }
        guard let cls = config.cls.token() else { fatalError("Missing `cls` processor configuration") }
        self.sep = sep
        self.cls = cls
        trimOffset = config.trimOffset.boolean(or: true)
        addPrefixSpace = config.addPrefixSpace.boolean(or: true)
    }

    func postProcess(tokens: [String], tokensPair: [String]?, addSpecialTokens: Bool = true) -> [String] {
        var outTokens = tokens
        var tokensPair = tokensPair
        if trimOffset {
            if addPrefixSpace {
                outTokens = outTokens.map { trimExtraSpaces(token: $0) }
                tokensPair = tokensPair?.map { trimExtraSpaces(token: $0) }
            } else {
                outTokens = outTokens.map { $0.trimmingCharacters(in: .whitespaces) }
                tokensPair = tokensPair?.map { $0.trimmingCharacters(in: .whitespaces) }
            }
        }

        outTokens = [cls.1] + outTokens + [sep.1]
        if let tokensPair, !tokensPair.isEmpty {
            // Yes, it adds another `sep`.
            // https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/roberta/hub_interface.py#L58-L65
            outTokens += [sep.1] + tokensPair + [sep.1]
        }

        return outTokens
    }

    /// Some tokens need one space around them
    /// https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/pre_tokenizers/byte_level.rs#L203-L235
    private func trimExtraSpaces(token: String) -> String {
        let prefixOffset = findPrefixIndex(text: token)
        let suffixOffset = findSuffixIndex(text: token)
        let prefixIndex = token.index(token.startIndex, offsetBy: prefixOffset)
        let suffixIndex = token.index(token.startIndex, offsetBy: token.count - suffixOffset)
        return String(token[prefixIndex..<suffixIndex])
    }

    private func findPrefixIndex(text: String) -> Int {
        guard !text.isEmpty, text.first!.isWhitespace else { return 0 }
        return text.prefix(while: { $0.isWhitespace }).count - 1
    }

    private func findSuffixIndex(text: String) -> Int {
        guard !text.isEmpty, text.last!.isWhitespace else { return 0 }
        return text.reversed().prefix(while: { $0.isWhitespace }).count - 1
    }
}

class BertProcessing: PostProcessor {
    private let sep: (UInt, String)
    private let cls: (UInt, String)

    public required init(config: Config) {
        guard let sep = config.sep.token() else { fatalError("Missing `sep` processor configuration") }
        guard let cls = config.cls.token() else { fatalError("Missing `cls` processor configuration") }
        self.sep = sep
        self.cls = cls
    }

    func postProcess(tokens: [String], tokensPair: [String]?, addSpecialTokens: Bool = true) -> [String] {
        guard addSpecialTokens else { return tokens + (tokensPair ?? []) }

        var outTokens = [cls.1] + tokens + [sep.1]
        if let tokensPair, !tokensPair.isEmpty {
            outTokens += tokensPair + [sep.1]
        }

        return outTokens
    }
}

class SequenceProcessing: PostProcessor {
    private let processors: [PostProcessor]

    public required init(config: Config) {
        guard let processorConfigs = config.processors.array() else {
            fatalError("Missing `processors` configuration")
        }

        processors = processorConfigs.compactMap { PostProcessorFactory.fromConfig(config: $0) }
    }

    func postProcess(tokens: [String], tokensPair: [String]?, addSpecialTokens: Bool = true) -> [String] {
        var currentTokens = tokens
        var currentTokensPair = tokensPair

        for processor in processors {
            let processed = processor.postProcess(tokens: currentTokens, tokensPair: currentTokensPair, addSpecialTokens: addSpecialTokens)
            currentTokens = processed
            currentTokensPair = nil // After the first processor, we no longer have a separate pair
        }

        return currentTokens
    }
}
