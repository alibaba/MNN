//
//  TokenLattice.swift
//
//
//  Created by Pedro Cuenca on 20240117.
//  Copyright © 2024 Hugging Face. All rights reserved.
//

/// Implements a TokenLattice to implement the Viterbi algorithm
/// We could make it generic so TokenLatticeNode stores an opaque type, but it's overkill right now.
/// Based on https://github.com/huggingface/tokenizers/blob/b58227c7f1ccf8b73ee2268354336da56d91e492/tokenizers/src/models/unigram/lattice.rs#L137
/// and https://github.com/xenova/transformers.js/blob/b07336d8f7ff57453cc164cc68aead2a79cbd57e/src/utils/data-structures.js#L269C28-L269C28
public struct TokenLattice {
    let sentence: String
    let bosTokenId: Int
    let eosTokenId: Int

    var nodes: [TokenLatticeNode] = []
    var beginNodes: [[TokenLatticeNode]]
    var endNodes: [[TokenLatticeNode]]

    var count: Int { sentence.count }

    init(sentence: String, bosTokenId: Int, eosTokenId: Int) {
        self.sentence = sentence
        self.bosTokenId = bosTokenId
        self.eosTokenId = eosTokenId

        beginNodes = Array(repeating: [], count: sentence.count + 1)
        endNodes = Array(repeating: [], count: sentence.count + 1)

        let bos = TokenLatticeNode(tokenId: bosTokenId, startOffset: 0, length: 0, score: 0)
        let eos = TokenLatticeNode(tokenId: eosTokenId, startOffset: sentence.count, length: 0, score: 0)

        nodes.append(bos)
        nodes.append(eos)

        beginNodes[sentence.count].append(eos)
        endNodes[0].append(bos)
    }
}

public extension TokenLattice {
    /// Insert a new token into the node lattice.
    ///
    ///  - Parameters:
    ///      - startOffset: Starting position of the token in the sentence.
    ///      - length: Number of characters in the token.
    ///      - score: Token score.
    ///      - tokenId: Token id in the tokenizer.
    mutating func insert(startOffset: Int, length: Int, score: Float, tokenId: Int) {
        let node = TokenLatticeNode(tokenId: tokenId, startOffset: startOffset, length: length, score: score)
        beginNodes[startOffset].append(node)
        endNodes[startOffset + length].append(node)
        nodes.append(node)
    }
}

extension TokenLattice {
    /// Implements the Viterbi algorithm to compute the most likely sequence of tokens.
    /// It's unfortunate that it can't be lazy or cached as the node arrays are not immutable.
    /// We could create another type that holds the nodes and use it as an immutable var  in TokenLattice.
    func viterbi() -> [TokenLatticeNode] {
        for offset in 0...count {
            guard beginNodes[offset].count > 0 else { return [] }

            for rnode in beginNodes[offset] {
                rnode.prev = nil
                var bestScore: Float = 0
                var bestNode: TokenLatticeNode?
                for lnode in endNodes[offset] {
                    let score = lnode.backtraceScore + rnode.score
                    if bestNode == nil || score > bestScore {
                        bestNode = lnode.clone()
                        bestScore = score
                    }
                }

                if bestNode != nil {
                    rnode.prev = bestNode
                    rnode.backtraceScore = bestScore
                }
            }
        }

        let root = beginNodes[count][0]
        guard let prev = root.prev else { return [] }

        // TODO: the reference implementations have a few more clones here: verify
        var result: [TokenLatticeNode] = []
        var node = prev // .clone()
        while node.prev != nil {
            result.append(node.clone())
            node = node.prev! // .clone()
        }
        return result.reversed()
    }

    /// Returns the substring of the sentence to be tokenized associated to the specified node
    ///
    /// - Parameters:
    ///     - node: The node defining the token to be extracted
    ///
    /// - Returns: A **Substring** – i.e., a reference to the original positions, not a copy of the characters.
    func piece(_ node: TokenLatticeNode) -> any StringProtocol {
        let start = sentence.index(sentence.startIndex, offsetBy: node.startOffset)
        let end = sentence.index(start, offsetBy: node.length)
        return sentence[start..<end]
    }
}

public extension TokenLattice {
    var tokens: [String] {
        viterbi().map { String(piece($0)) }
    }

    var tokenIds: [Int] {
        viterbi().map { $0.tokenId }
    }
}

class TokenLatticeNode {
    let tokenId: Int
    let startOffset: Int
    let length: Int
    let score: Float

    var prev: TokenLatticeNode?
    var backtraceScore: Float = 0

    init(tokenId: Int, startOffset: Int, length: Int, score: Float, prev: TokenLatticeNode? = nil, backtraceScore: Float = 0) {
        self.tokenId = tokenId
        self.startOffset = startOffset
        self.length = length
        self.score = score
        self.prev = prev
        self.backtraceScore = backtraceScore
    }
}

extension TokenLatticeNode {
    /// This is a reference type because structs can't contain references to the same type
    /// We could implement NSCopying, but frankly I don't see the point
    func clone() -> TokenLatticeNode {
        TokenLatticeNode(tokenId: tokenId, startOffset: startOffset, length: length, score: score, prev: prev, backtraceScore: backtraceScore)
    }
}

extension TokenLatticeNode: CustomStringConvertible {
    var description: String {
        "TokenLatticeNode(tokenId: \(tokenId), startOffset: \(startOffset), length: \(length), score: \(score), prev: \(prev != nil), backtraceScore: \(backtraceScore)"
    }
}
