//
//  UnigramTokenizer.swift
//
//
//  Created by Pedro Cuenca on 20240131.
//  Copyright © 2024 Hugging Face. All rights reserved.
//

import Foundation
import Hub

class UnigramTokenizer: PreTrainedTokenizerModel {
    struct SentencePieceToken {
        var token: String
        var score: Float
    }

    let vocab: [SentencePieceToken]

    let unknownPiece: SentencePieceToken
    var unknownTokenScore: Float { unknownPiece.score }

    public let unknownTokenId: Int?
    public var unknownToken: String? { unknownPiece.token }

    let minScore: Float
    let tokensToIds: [NSString: Int]

    let bosToken: String? = " "
    let bosTokenId: Int?
    let eosToken: String?
    let eosTokenId: Int?

    /// Hardcoded in Unigram tokenizers
    let fuseUnknownTokens: Bool = true

    private let trie: Trie<Character>

    required init(tokenizerConfig: Config, tokenizerData: Config, addedTokens: [String: Int]) throws {
        guard let configVocab = tokenizerData.model.vocab.array() else {
            throw TokenizerError.missingVocab
        }

        vocab = try configVocab.map { piece in
            let tuple = piece.array(or: [])

            guard let token = tuple.first?.string(),
                  let scoreValue = tuple.last
            else {
                throw TokenizerError.malformedVocab
            }

            let score: Float
            if let floatScore = scoreValue.floating() {
                score = floatScore
            } else if let numberScore = scoreValue.integer() {
                score = Float(numberScore)

            } else {
                throw TokenizerError.malformedVocab
            }

            return SentencePieceToken(token: token, score: score)
        }

        minScore = vocab.reduce(999) { partial, token in
            min(partial, token.score)
        }

        guard let unknownTokenId = tokenizerData.model["unkId"].integer() else { throw TokenizerError.malformedVocab }
        self.unknownTokenId = unknownTokenId
        unknownPiece = SentencePieceToken(token: vocab[unknownTokenId].token, score: minScore - 10)

        tokensToIds = Dictionary(uniqueKeysWithValues: vocab.map { $0.token as NSString }.enumerated().map { ($1, $0) })
        bosTokenId = tokensToIds[bosToken! as NSString] // May be nil

        eosToken = tokenizerConfig.eosToken.string()
        eosTokenId = eosToken == nil ? nil : tokensToIds[eosToken! as NSString]

        trie = Trie()
        trie.append(contentsOf: vocab.map { $0.token })
    }

    func convertTokenToId(_ token: String) -> Int? {
        tokensToIds[token as NSString] ?? unknownTokenId
    }

    func convertIdToToken(_ id: Int) -> String? {
        vocab[id].token
    }

    func tokenize(text: String) -> [String] {
        var lattice = TokenLattice(sentence: text, bosTokenId: bosTokenId ?? 0, eosTokenId: eosTokenId ?? 0)

        // Populate nodes
        let sentence = lattice.sentence
        var beginPos = 0
        while beginPos < sentence.count {
            let mblen = 1
            var hasSingleNode = false

            let beginIndex = sentence.index(sentence.startIndex, offsetBy: beginPos)
            for token in trie.commonPrefixSearchIterator(sentence[beginIndex...]).map({ String($0) }) {
                guard let tokenId = tokensToIds[token as NSString] else { fatalError("Token not in vocab: \(token)") }
                let tokenScore = vocab[tokenId].score
                lattice.insert(startOffset: beginPos, length: token.count, score: tokenScore, tokenId: tokenId)
                if !hasSingleNode, token.count == mblen {
                    hasSingleNode = true
                }
            }
            if !hasSingleNode {
                lattice.insert(startOffset: beginPos, length: mblen, score: unknownTokenScore, tokenId: unknownTokenId ?? 0)
            }
            beginPos += mblen
        }

        return lattice.tokens
    }
}
