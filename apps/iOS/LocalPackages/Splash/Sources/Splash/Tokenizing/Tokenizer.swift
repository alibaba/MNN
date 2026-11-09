/**
 *  Splash
 *  Copyright (c) John Sundell 2018
 *  MIT license - see LICENSE.md
 */

import Foundation

internal struct Tokenizer {
    func segmentsByTokenizing(_ code: String,
                              using grammar: Grammar) -> AnySequence<Segment> {
        return AnySequence<Segment> {
            Buffer(iterator: Iterator(code: code, grammar: grammar))
        }
    }
}

private extension Tokenizer {
    struct Buffer: IteratorProtocol {
        private var iterator: Iterator
        private var nextSegment: Segment?

        init(iterator: Iterator) {
            self.iterator = iterator
        }

        mutating func next() -> Segment? {
            var segment = nextSegment ?? iterator.controlledIterate(depth: 32)
            nextSegment = iterator.controlledIterate(depth: 32)
            segment?.tokens.next = nextSegment?.tokens.current
            return segment
        }
    }

    struct Iterator: IteratorProtocol {
        struct Component {
            enum Kind {
                case token
                case delimiter
                case whitespace
                case newline
            }

            let character: Character
            let kind: Kind
        }

        private let code: String
        private let grammar: Grammar
        private var index: String.Index?
        private var tokenCounts = [String: Int]()
        private var allTokens = [String]()
        private var lineTokens = [String]()
        private var segments: (current: Segment?, previous: Segment?)

        init(code: String, grammar: Grammar) {
            self.code = code
            self.grammar = grammar
            segments = (nil, nil)
        }

        mutating func next() -> Segment? {
            controlledIterate(depth: 32)
        }

        mutating func controlledIterate(depth: Int) -> Segment? {
            guard depth >= 0 else { return nil }

            let nextIndex = makeNextIndex()

            guard nextIndex != code.endIndex else {
                let segment = segments.current
                segments.current = nil
                return segment
            }

            index = nextIndex
            let component = makeComponent(at: nextIndex)

            switch component.kind {
            case .token, .delimiter:
                guard var segment = segments.current else {
                    segments.current = makeSegment(with: component, at: nextIndex)
                    return controlledIterate(depth: depth - 1)
                }

                guard segment.trailingWhitespace == nil,
                      component.isDelimiter == segment.currentTokenIsDelimiter
                else {
                    return finish(segment, with: component, at: nextIndex)
                }

                if component.isDelimiter {
                    let previousCharacter = segment.tokens.current.last!
                    let shouldMerge = grammar.isDelimiter(previousCharacter,
                                                          mergableWith: component.character)

                    guard shouldMerge else {
                        return finish(segment, with: component, at: nextIndex)
                    }
                }

                segment.tokens.current.append(component.character)
                segments.current = segment
                return controlledIterate(depth: depth - 1)
            case .whitespace, .newline:
                guard var segment = segments.current else {
                    var segment = makeSegment(with: component, at: nextIndex)
                    segment.trailingWhitespace = component.token
                    segment.isLastOnLine = component.isNewline
                    segments.current = segment
                    return controlledIterate(depth: depth - 1)
                }

                if var existingWhitespace = segment.trailingWhitespace {
                    existingWhitespace.append(component.character)
                    segment.trailingWhitespace = existingWhitespace
                } else {
                    segment.trailingWhitespace = component.token
                }

                if component.isNewline {
                    segment.isLastOnLine = true
                }

                segments.current = segment
                return controlledIterate(depth: depth - 1)
            }
        }

        private func makeNextIndex() -> String.Index {
            guard let index = index else {
                return code.startIndex
            }

            return code.index(after: index)
        }

        private func makeComponent(at index: String.Index) -> Component {
            func kind(for character: Character) -> Component.Kind {
                if character.isWhitespace {
                    return .whitespace
                }

                if character.isNewline {
                    return .newline
                }

                if grammar.delimiters.contains(character) {
                    return .delimiter
                }

                return .token
            }

            let character = code[index]

            return Component(
                character: character,
                kind: kind(for: character)
            )
        }

        private func makeSegment(with component: Component, at index: String.Index) -> Segment {
            let tokens = Segment.Tokens(
                all: allTokens,
                counts: tokenCounts,
                onSameLine: lineTokens,
                previous: segments.current?.tokens.current,
                current: component.token,
                next: nil
            )

            return Segment(
                prefix: code[..<index],
                tokens: tokens,
                trailingWhitespace: nil,
                currentTokenIsDelimiter: component.isDelimiter,
                isLastOnLine: false
            )
        }

        private mutating func finish(_ segment: Segment,
                                     with component: Component,
                                     at index: String.Index) -> Segment {
            var count = tokenCounts[segment.tokens.current] ?? 0
            count += 1
            tokenCounts[segment.tokens.current] = count

            allTokens.append(segment.tokens.current)

            if segment.isLastOnLine {
                lineTokens = []
            } else {
                lineTokens.append(segment.tokens.current)
            }

            segments.previous = segment
            segments.current = makeSegment(with: component, at: index)

            return segment
        }
    }
}

extension Tokenizer.Iterator.Component {
    var token: String {
        return String(character)
    }

    var isDelimiter: Bool {
        switch kind {
        case .token, .whitespace, .newline:
            return false
        case .delimiter:
            return true
        }
    }

    var isNewline: Bool {
        switch kind {
        case .token, .whitespace, .delimiter:
            return false
        case .newline:
            return true
        }
    }
}

private extension Character {
    var isWhitespace: Bool {
        return CharacterSet.whitespaces.contains(self)
    }

    var isNewline: Bool {
        return CharacterSet.newlines.contains(self)
    }
}
