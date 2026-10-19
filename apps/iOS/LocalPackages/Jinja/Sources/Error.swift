//
//  Error.swift
//
//
//  Created by John Mai on 2024/3/20.
//

import Foundation

enum JinjaError: Error, LocalizedError, Equatable {
    case syntax(String)
    case parser(String)
    case runtime(String)
    case todo(String)
    case syntaxNotSupported(String)

    var errorDescription: String? {
        switch self {
        case .syntax(let message): return "Syntax error: \(message)"
        case .parser(let message): return "Parser error: \(message)"
        case .runtime(let message): return "Runtime error: \(message)"
        case .todo(let message): return "Todo error: \(message)"
        case .syntaxNotSupported(let string): return "Syntax not supported: \(string)"
        }
    }

    var id: String {
        errorDescription ?? ""
    }
}
