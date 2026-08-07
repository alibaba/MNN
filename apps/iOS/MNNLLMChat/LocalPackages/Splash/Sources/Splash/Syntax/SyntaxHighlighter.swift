/**
 *  Splash
 *  Copyright (c) John Sundell 2018
 *  MIT license - see LICENSE.md
 */

import Foundation

/// This class acts as the main API entry point for Splash. Using it,
/// any code string can be highlighted to any desired format, using
/// any language grammar. Per default, Swift's langauge grammar is used.
/// To initialize this class, pass the desired output format, such as
/// `AttributedStringOutputFormat` or `HTMLOutputFormat`, or a custom
/// implementation. One syntax highlighter may be reused multiple times.
public struct SyntaxHighlighter<Format: OutputFormat> {
    private let format: Format
    private let grammar: Grammar
    private let tokenizer = Tokenizer()

    /// Initialize an instance with the desired output format.
    /// If no grammar is passed, then Swift's grammar is used.
    public init(format: Format, grammar: Grammar = SwiftGrammar()) {
        self.format = format
        self.grammar = grammar
    }

    /// Highlight the given code, returning output as specified by the
    /// syntax highlighter's `Format`.
    public func highlight(_ code: String) -> Format.Builder.Output {
        var builder = format.makeBuilder()
        var state: (token: String, tokenType: TokenType?)?

        func handle(_ token: String, ofType type: TokenType?, trailingWhitespace: String?) {
            guard let whitespace = trailingWhitespace else {
                state = (token, type)
                return
            }

            builder.addToken(token, ofType: type)
            builder.addWhitespace(whitespace)
            state = nil
        }

        for segment in tokenizer.segmentsByTokenizing(code, using: grammar) {
            let token = segment.tokens.current
            let whitespace = segment.trailingWhitespace

            guard !token.isEmpty else {
                whitespace.map { builder.addWhitespace($0) }
                continue
            }

            let tokenType = typeOfToken(in: segment)

            guard var currentState = state else {
                handle(token, ofType: tokenType, trailingWhitespace: whitespace)
                continue
            }

            guard currentState.tokenType == tokenType else {
                builder.addToken(currentState.token, ofType: currentState.tokenType)
                handle(token, ofType: tokenType, trailingWhitespace: whitespace)
                continue
            }

            currentState.token.append(token)
            handle(currentState.token, ofType: tokenType, trailingWhitespace: whitespace)
        }

        if let lastState = state {
            builder.addToken(lastState.token, ofType: lastState.tokenType)
        }

        return builder.build()
    }

    private func typeOfToken(in segment: Segment) -> TokenType? {
        let rule = grammar.syntaxRules.first { $0.matches(segment) }
        return rule?.tokenType
    }
}

private extension OutputBuilder {
    mutating func addToken(_ token: String, ofType type: TokenType?) {
        if let type = type {
            addToken(token, ofType: type)
        } else {
            addPlainText(token)
        }
    }
}
