//
//  Lexer.swift
//
//
//  Created by John Mai on 2024/3/20.
//

import Foundation

enum TokenType: String {
    case text = "Text"

    case numericLiteral = "NumericLiteral"
    case booleanLiteral = "BooleanLiteral"
    case nullLiteral = "NullLiteral"
    case stringLiteral = "StringLiteral"
    case identifier = "Identifier"
    case equals = "Equals"
    case openParen = "OpenParen"
    case closeParen = "CloseParen"
    case openStatement = "OpenStatement"
    case closeStatement = "CloseStatement"
    case openExpression = "OpenExpression"
    case closeExpression = "CloseExpression"
    case openSquareBracket = "OpenSquareBracket"
    case closeSquareBracket = "CloseSquareBracket"
    case openCurlyBracket = "OpenCurlyBracket"
    case closeCurlyBracket = "CloseCurlyBracket"
    case comma = "Comma"
    case dot = "Dot"
    case colon = "Colon"
    case pipe = "Pipe"

    case callOperator = "CallOperator"
    case additiveBinaryOperator = "AdditiveBinaryOperator"
    case multiplicativeBinaryOperator = "MultiplicativeBinaryOperator"
    case comparisonBinaryOperator = "ComparisonBinaryOperator"
    case unaryOperator = "UnaryOperator"

    case set = "Set"
    case `if` = "If"
    case `for` = "For"
    case `in` = "In"
    case `is` = "Is"
    case notIn = "NotIn"
    case `else` = "Else"
    case endIf = "EndIf"
    case elseIf = "ElseIf"
    case endFor = "EndFor"
    case and = "And"
    case or = "Or"
    case not = "Not"
    case macro = "Macro"
    case endMacro = "EndMacro"
}

struct Token: Equatable {
    var value: String
    var type: TokenType
}

let keywords: [String: TokenType] = [
    "set": .set,
    "for": .for,
    "in": .in,
    "is": .is,
    "if": .if,
    "else": .else,
    "endif": .endIf,
    "elif": .elseIf,
    "endfor": .endFor,
    "and": .and,
    "or": .or,
    "not": .not,
    "macro": .macro,
    "endmacro": .endMacro,
    // Literals
    "true": .booleanLiteral,
    "false": .booleanLiteral,
    "none": .nullLiteral,
]

func isWord(char: String) -> Bool {
    char.range(of: #"\w"#, options: .regularExpression) != nil
}

func isInteger(char: String) -> Bool {
    char.range(of: #"^[0-9]+$"#, options: .regularExpression) != nil
}

func isWhile(char: String) -> Bool {
    char.range(of: #"\s"#, options: .regularExpression) != nil
}

let orderedMappingTable: [(String, TokenType)] = [
    ("{%", .openStatement),
    ("%}", .closeStatement),
    ("{{", .openExpression),
    ("}}", .closeExpression),
    ("(", .openParen),
    (")", .closeParen),
    ("{", .openCurlyBracket),
    ("}", .closeCurlyBracket),
    ("[", .openSquareBracket),
    ("]", .closeSquareBracket),
    (",", .comma),
    (".", .dot),
    (":", .colon),
    ("|", .pipe),
    ("<=", .comparisonBinaryOperator),
    (">=", .comparisonBinaryOperator),
    ("==", .comparisonBinaryOperator),
    ("!=", .comparisonBinaryOperator),
    ("<", .comparisonBinaryOperator),
    (">", .comparisonBinaryOperator),
    ("+", .additiveBinaryOperator),
    ("-", .additiveBinaryOperator),
    ("~", .additiveBinaryOperator),
    ("*", .multiplicativeBinaryOperator),
    ("/", .multiplicativeBinaryOperator),
    ("%", .multiplicativeBinaryOperator),
    ("=", .equals),
]

let escapeCharacters: [String: String] = [
    "n": "\n",
    "t": "\t",
    "r": "\r",
    "b": "\u{0008}",
    "f": "\u{000C}",
    "v": "\u{000B}",
    "'": "'",
    "\"": "\"",
    "\\": "\\",
]

struct PreprocessOptions {
    var trimBlocks: Bool?
    var lstripBlocks: Bool?
}

func preprocess(template: String, options: PreprocessOptions = PreprocessOptions()) -> String {
    var template = template
    if template.hasSuffix("\n") {
        template.removeLast()
    }
    template = template.replacing(#/{#.*?#}/#, with: "{##}")
    if options.lstripBlocks == true {
        template = template.replacing(#/(?m)^[ \t]*({[#%])/#, with: { $0.output.1 })
    }
    if options.trimBlocks == true {
        template = template.replacing(#/([#%]})\n/#, with: { $0.output.1 })
    }
    return
        template
        .replacing(#/{##}/#, with: "")
        .replacing(#/-%}\s*/#, with: "%}")
        .replacing(#/\s*{%-/#, with: "{%")
        .replacing(#/-}}\s*/#, with: "}}")
        .replacing(#/\s*{{-/#, with: "{{")
}

func tokenize(_ source: String, options: PreprocessOptions = PreprocessOptions()) throws -> [Token] {
    var tokens: [Token] = []
    let src = preprocess(template: source, options: options)
    var cursorPosition = 0

    @discardableResult
    func consumeWhile(predicate: (String) -> Bool) throws -> String {
        var str = ""
        while cursorPosition < src.count, predicate(String(src[cursorPosition])) {
            if src[cursorPosition] == "\\" {
                cursorPosition += 1
                if cursorPosition >= src.count {
                    throw JinjaError.syntax("Unexpected end of input")
                }
                let escaped = String(src[cursorPosition])
                cursorPosition += 1
                guard let unescaped = escapeCharacters[escaped] else {
                    throw JinjaError.syntax("Unexpected escaped character: \(escaped)")
                }
                str.append(unescaped)
                continue
            }
            str.append(String(src[cursorPosition]))
            cursorPosition += 1
            if cursorPosition >= src.count {
                throw JinjaError.syntax("Unexpected end of input")
            }
        }
        return str
    }

    main: while cursorPosition < src.count {
        let lastTokenType = tokens.last?.type
        if lastTokenType == nil || lastTokenType == .closeStatement || lastTokenType == .closeExpression {
            var text = ""

            while cursorPosition < src.count,
                !(src[cursorPosition] == "{" && (src[cursorPosition + 1] == "%" || src[cursorPosition + 1] == "{"))
            {
                text.append(src[cursorPosition])
                cursorPosition += 1
            }

            if !text.isEmpty {
                tokens.append(Token(value: text, type: .text))
                continue
            }
        }
        try consumeWhile(predicate: isWhile)
        let char = String(src[cursorPosition])
        if char == "-" || char == "+" {
            let lastTokenType = tokens.last?.type
            if lastTokenType == .text || lastTokenType == nil {
                throw JinjaError.syntax("Unexpected character: \(char)")
            }
            switch lastTokenType {
            case .identifier,
                .numericLiteral,
                .booleanLiteral,
                .nullLiteral,
                .stringLiteral,
                .closeParen,
                .closeSquareBracket:
                break
            default:
                cursorPosition += 1
                let num = try consumeWhile(predicate: isInteger)
                tokens.append(Token(value: "\(char)\(num)", type: num.isEmpty ? .unaryOperator : .numericLiteral))
                continue
            }
        }
        for (char, token) in orderedMappingTable {
            let slice = src.slice(start: cursorPosition, end: cursorPosition + char.count)
            if slice == char {
                tokens.append(Token(value: char, type: token))
                cursorPosition += char.count
                continue main
            }
        }
        if char == "'" || char == "\"" {
            cursorPosition += 1
            let str = try consumeWhile { str in
                str != char
            }
            tokens.append(Token(value: str, type: .stringLiteral))
            cursorPosition += 1
            continue
        }
        if isInteger(char: char) {
            let num = try consumeWhile(predicate: isInteger)
            tokens.append(Token(value: num, type: .numericLiteral))
            continue
        }
        if isWord(char: char) {
            let word = try consumeWhile(predicate: isWord)
            let type: TokenType = keywords.contains(where: { $0.key == word }) ? keywords[word]! : .identifier
            if type == .in, tokens.last?.type == .not {
                _ = tokens.popLast()
                tokens.append(Token(value: "not in", type: .notIn))
            } else {
                tokens.append(Token(value: word, type: type))
            }
            continue
        }
        throw JinjaError.syntax("Unexpected character: \(char)")
    }
    return tokens
}
