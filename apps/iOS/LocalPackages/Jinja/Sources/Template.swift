//
//  Template.swift
//
//
//  Created by John Mai on 2024/3/23.
//

import Foundation

public struct Template {
    var parsed: Program

    public init(_ template: String) throws {
        let tokens = try tokenize(template, options: PreprocessOptions(trimBlocks: true, lstripBlocks: true))
        self.parsed = try parse(tokens: tokens)
    }

    public func render(_ items: [String: Any?]) throws -> String {
        let env = Environment()

        try env.set(name: "false", value: false)
        try env.set(name: "true", value: true)
        try env.set(name: "none", value: NullValue())
        try env.set(
            name: "raise_exception",
            value: { (args: String) throws in
                throw JinjaError.runtime("\(args)")
            }
        )
        try env.set(name: "range", value: range)

        for (key, value) in items {
            if let value {
                try env.set(name: key, value: value)
            }
        }

        let interpreter = Interpreter(env: env)
        let result = try interpreter.run(program: self.parsed) as! StringValue

        return result.value
    }
}
