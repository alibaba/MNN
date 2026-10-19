//
//  BertTokenizer.swift
//  CoreMLBert
//
//  Created by Julien Chaumond on 27/06/2019.
//  Copyright © 2019 Hugging Face. All rights reserved.
//

import Foundation
import Hub

public class BertTokenizer {
    private let basicTokenizer: BasicTokenizer
    private let wordpieceTokenizer: WordpieceTokenizer
    private let maxLen = 512
    private let tokenizeChineseChars: Bool

    private let vocab: [String: Int]
    private let ids_to_tokens: [Int: String]

    public var bosToken: String?
    public var bosTokenId: Int?
    public var eosToken: String?
    public var eosTokenId: Int?

    public let fuseUnknownTokens: Bool

    public init(
        vocab: [String: Int],
        merges: [String]?,
        tokenizeChineseChars: Bool = true,
        bosToken: String? = nil,
        eosToken: String? = nil,
        fuseUnknownTokens: Bool = false,
        doLowerCase: Bool = true
    ) {
        self.vocab = vocab
        ids_to_tokens = Utils.invert(vocab)
        basicTokenizer = BasicTokenizer(doLowerCase: doLowerCase)
        wordpieceTokenizer = WordpieceTokenizer(vocab: self.vocab)
        self.tokenizeChineseChars = tokenizeChineseChars
        self.bosToken = bosToken
        bosTokenId = bosToken == nil ? nil : vocab[bosToken!]
        self.eosToken = eosToken
        eosTokenId = eosToken == nil ? nil : vocab[eosToken!]
        self.fuseUnknownTokens = fuseUnknownTokens
    }

    public required convenience init(tokenizerConfig: Config, tokenizerData: Config, addedTokens: [String: Int]) throws {
        guard let vocab = tokenizerData.model.vocab.dictionary() else {
            throw TokenizerError.missingVocab
        }

        let merges: [String]? = tokenizerData.model.merges.get()
        let tokenizeChineseChars = tokenizerConfig.handleChineseChars.boolean(or: true)
        let eosToken = tokenizerConfig.eosToken.string()
        let bosToken = tokenizerConfig.bosToken.string()
        let fuseUnknown = tokenizerConfig.fuseUnk.boolean(or: false)
        let doLowerCase = tokenizerConfig.doLowerCase.boolean(or: true)

        var vocabulary = vocab.reduce(into: [String: Int]()) { result, element in
            if let val = element.value.integer() {
                result[element.key.string] = val
            }
        }
        if let pairs = tokenizerData.addedTokens.array()?.reduce(into: [String: Int](), { result, element in
            guard let val = element["id"].integer() else { return }
            guard let key = element["content"].string() else { return }

            result[key] = val
        }) {
            vocabulary.merge(pairs, uniquingKeysWith: { $1 })
        }

        vocabulary.merge(addedTokens, uniquingKeysWith: { $1 })

        self.init(
            vocab: vocabulary, merges: merges, tokenizeChineseChars: tokenizeChineseChars, bosToken: bosToken, eosToken: eosToken,
            fuseUnknownTokens: fuseUnknown, doLowerCase: doLowerCase
        )
    }

    public func tokenize(text: String) -> [String] {
        let text = tokenizeChineseCharsIfNeed(text)
        var tokens: [String] = []
        for token in basicTokenizer.tokenize(text: text) {
            for subToken in wordpieceTokenizer.tokenize(word: token) {
                tokens.append(subToken)
            }
        }
        return tokens
    }

    private func convertTokensToIds(tokens: [String]) throws -> [Int] {
        if tokens.count > maxLen {
            throw TokenizerError.tooLong(
                """
                Token indices sequence length is longer than the specified maximum
                sequence length for this BERT model (\(tokens.count) > \(maxLen). Running this
                sequence through BERT will result in indexing errors".format(len(ids), self.max_len)
                """
            )
        }
        return tokens.compactMap { vocab[$0] }
    }

    /// Main entry point
    func tokenizeToIds(text: String) -> [Int] {
        try! convertTokensToIds(tokens: tokenize(text: text))
    }

    func tokenToId(token: String) -> Int {
        vocab[token]!
    }

    /// Un-tokenization: get tokens from tokenIds
    func unTokenize(tokens: [Int]) -> [String] {
        tokens.compactMap { ids_to_tokens[$0] }
    }

    /// Un-tokenization:
    func convertWordpieceToBasicTokenList(_ wordpieceTokenList: [String]) -> String {
        var tokenList: [String] = []
        var individualToken = ""
        for token in wordpieceTokenList {
            if token.starts(with: "##") {
                individualToken += String(token.suffix(token.count - 2))
            } else {
                if individualToken.count > 0 {
                    tokenList.append(individualToken)
                }

                individualToken = token
            }
        }

        tokenList.append(individualToken)

        return tokenList.joined(separator: " ")
    }

    private func tokenizeChineseCharsIfNeed(_ text: String) -> String {
        guard tokenizeChineseChars else {
            return text
        }

        return text.map { c in
            if let scalar = c.unicodeScalars.first, Utils.isChineseChar(scalar) {
                " \(c) "
            } else {
                "\(c)"
            }
        }.joined()
    }
}

extension BertTokenizer: PreTrainedTokenizerModel {
    public var unknownToken: String? { wordpieceTokenizer.unkToken }
    public var unknownTokenId: Int? { vocab[unknownToken!] }

    func encode(text: String) -> [Int] { tokenizeToIds(text: text) }

    func decode(tokens: [Int]) -> String {
        let tokens = unTokenize(tokens: tokens)
        return convertWordpieceToBasicTokenList(tokens)
    }

    public func convertTokenToId(_ token: String) -> Int? {
        vocab[token] ?? unknownTokenId
    }

    public func convertIdToToken(_ id: Int) -> String? {
        ids_to_tokens[id]
    }
}

class BasicTokenizer {
    let doLowerCase: Bool

    init(doLowerCase: Bool = true) {
        self.doLowerCase = doLowerCase
    }

    let neverSplit = [
        "[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]",
    ]

    func maybeStripAccents(_ text: String) -> String {
        guard doLowerCase else { return text }
        return text.folding(options: .diacriticInsensitive, locale: nil)
    }

    func maybeLowercase(_ text: String) -> String {
        guard doLowerCase else { return text }
        return text.lowercased()
    }

    func tokenize(text: String) -> [String] {
        let splitTokens = maybeStripAccents(text).components(separatedBy: NSCharacterSet.whitespaces)
        let tokens = splitTokens.flatMap { (token: String) -> [String] in
            if neverSplit.contains(token) {
                return [token]
            }
            var toks: [String] = []
            var currentTok = ""
            for c in maybeLowercase(token) {
                if !c.isExtendedPunctuation {
                    currentTok += String(c)
                } else if currentTok.count > 0 {
                    toks.append(currentTok)
                    toks.append(String(c))
                    currentTok = ""
                } else {
                    toks.append(String(c))
                }
            }
            if currentTok.count > 0 {
                toks.append(currentTok)
            }
            return toks
        }
        return tokens
    }
}

extension Character {
    /// https://github.com/huggingface/transformers/blob/8c1b5d37827a6691fef4b2d926f2d04fb6f5a9e3/src/transformers/tokenization_utils.py#L367
    var isExtendedPunctuation: Bool {
        if isPunctuation { return true }
        if let value = unicodeScalars.first?.value {
            switch value {
            case 33...47: return true
            case 58...64: return true
            case 91...96: return true
            case 123...126: return true
            default: return false
            }
        }
        return false
    }
}

class WordpieceTokenizer {
    let unkToken = "[UNK]"
    private let maxInputCharsPerWord = 100
    private let vocab: [String: Int]

    init(vocab: [String: Int]) {
        self.vocab = vocab
    }

    /// `word`: A single token.
    /// Warning: this differs from the `pytorch-transformers` implementation.
    /// This should have already been passed through `BasicTokenizer`.
    func tokenize(word: String) -> [String] {
        if word.count > maxInputCharsPerWord {
            return [unkToken]
        }
        var outputTokens: [String] = []
        var isBad = false
        var start = 0
        var subTokens: [String] = []
        while start < word.count {
            var end = word.count
            var cur_substr: String?
            while start < end {
                var substr = Utils.substr(word, start..<end)!
                if start > 0 {
                    substr = "##\(substr)"
                }
                if vocab[substr] != nil {
                    cur_substr = substr
                    break
                }
                end -= 1
            }
            if cur_substr == nil {
                isBad = true
                break
            }
            subTokens.append(cur_substr!)
            start = end
        }
        if isBad {
            outputTokens.append(unkToken)
        } else {
            outputTokens.append(contentsOf: subTokens)
        }
        return outputTokens
    }
}
