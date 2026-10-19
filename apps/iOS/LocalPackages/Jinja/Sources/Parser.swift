//
//  Parser.swift
//
//
//  Created by John Mai on 2024/3/21.
//

import Foundation
import OrderedCollections

func parse(tokens: [Token]) throws -> Program {
    var program = Program()
    var current = 0

    @discardableResult
    func expect(type: TokenType, error: String) throws -> Token {
        let prev = tokens[current]
        current += 1
        if prev.type != type {
            throw JinjaError.parser("Parser Error: \(error). \(prev.type) != \(type).")
        }

        return prev
    }

    func parseArgumentsList() throws -> [Expression] {
        var args: [Expression] = []
        while !typeof(.closeParen) {
            var argument = try parseExpression()
            if typeof(.equals) {
                current += 1  // consume equals
                if let identifier = argument as? Identifier {
                    let value = try parseExpression()
                    argument = KeywordArgumentExpression(key: identifier, value: value)
                } else {
                    throw JinjaError.syntax("Expected identifier for keyword argument")
                }
            }
            args.append(argument)
            if typeof(.comma) {
                current += 1  // consume comma
            }
        }
        return args
    }

    func parseArgs() throws -> [Expression] {
        try expect(type: .openParen, error: "Expected opening parenthesis for arguments list")
        let args = try parseArgumentsList()
        try expect(type: .closeParen, error: "Expected closing parenthesis for arguments list")
        return args
    }

    func parseText() throws -> StringLiteral {
        try StringLiteral(value: expect(type: .text, error: "Expected text token").value)
    }

    func parseCallExpression(callee: Expression) throws -> Expression {
        let args = try parseArgs()
        var expression: Expression = CallExpression(callee: callee, args: args)
        // Handle potential array indexing after method call
        if typeof(.openSquareBracket) {
            expression = MemberExpression(
                object: expression,
                property: try parseMemberExpressionArgumentsList(),
                computed: true
            )
        }
        // Handle potential chained method calls
        if typeof(.openParen) {
            expression = try parseCallExpression(callee: expression)
        }
        return expression
    }

    func parseMemberExpressionArgumentsList() throws -> Expression {
        var slices: [Expression?] = []
        var isSlice = false
        while !typeof(.closeSquareBracket) {
            if typeof(.colon) {
                slices.append(nil)
                current += 1  // consume colon
                isSlice = true
            } else {
                // Handle negative numbers as indices
                if typeof(.additiveBinaryOperator) && tokens[current].value == "-" {
                    current += 1  // consume the minus sign
                    if typeof(.numericLiteral) {
                        let num = tokens[current].value
                        current += 1
                        slices.append(NumericLiteral(value: -Int(num)!))
                    } else {
                        throw JinjaError.syntax("Expected number after minus sign in array index")
                    }
                } else {
                    slices.append(try parseExpression())
                }
                if typeof(.colon) {
                    current += 1  // consume colon
                    isSlice = true
                }
            }
        }
        if slices.isEmpty {
            throw JinjaError.syntax("Expected at least one argument for member/slice expression")
        }
        if isSlice {
            if slices.count > 3 {
                throw JinjaError.syntax("Expected 0-3 arguments for slice expression")
            }
            return SliceExpression(
                start: slices[0],
                stop: slices.count > 1 ? slices[1] : nil,
                step: slices.count > 2 ? slices[2] : nil
            )
        }
        return slices[0]!  // normal member expression
    }

    func parseMemberExpression() throws -> Expression {
        var object = try parsePrimaryExpression()
        while typeof(.dot) || typeof(.openSquareBracket) {
            let operation = tokens[current]
            current += 1
            var property: Expression
            let computed = operation.type != .dot
            if computed {
                property = try parseMemberExpressionArgumentsList()
                try expect(type: .closeSquareBracket, error: "Expected closing square bracket")
            } else {
                property = try parsePrimaryExpression()
                if !(property is Identifier) {
                    throw JinjaError.syntax("Expected identifier following dot operator")
                }
                // Handle method calls
                if typeof(.openParen) {
                    let methodCall = CallExpression(
                        callee: MemberExpression(object: object, property: property, computed: false),
                        args: try parseArgs()
                    )
                    // Handle array indexing after method call
                    if typeof(.openSquareBracket) {
                        current += 1  // consume [
                        let index = try parseExpression()
                        try expect(type: .closeSquareBracket, error: "Expected closing square bracket")
                        object = MemberExpression(object: methodCall, property: index, computed: true)
                        continue
                    }
                    object = methodCall
                    continue
                }
            }
            object = MemberExpression(
                object: object,
                property: property,
                computed: computed
            )
        }
        return object
    }

    func parseCallMemberExpression() throws -> Expression {
        let member = try parseMemberExpression()
        if typeof(.openParen) {
            return try parseCallExpression(callee: member)
        }
        return member
    }

    func parseFilterExpression() throws -> Expression {
        var operand = try parseCallMemberExpression()
        while typeof(.pipe) {
            current += 1  // consume pipe
            guard let filterName = try parsePrimaryExpression() as? Identifier else {
                throw JinjaError.syntax("Expected identifier for the filter")
            }
            var args: [Expression] = []
            var kwargs: [KeywordArgumentExpression] = []
            var dyn_args: Expression?
            var dyn_kwargs: Expression?
            if typeof(.openParen) {
                // Handle filter with arguments
                (args, kwargs, dyn_args, dyn_kwargs) = try parseCallArgs()
            }
            operand = FilterExpression(
                operand: operand,
                filter: filterName,
                args: args,
                kwargs: kwargs,
                dyn_args: dyn_args,
                dyn_kwargs: dyn_kwargs
            )
        }
        return operand
    }

    func parseCallArgs() throws -> (
        [Expression], [KeywordArgumentExpression], Expression?, Expression?
    ) {
        try expect(type: .openParen, error: "Expected opening parenthesis for arguments list")
        var args: [Expression] = []
        var kwargs: [KeywordArgumentExpression] = []
        var dynArgs: Expression?
        var dynKwargs: Expression?
        var requireComma = false
        while !typeof(.closeParen) {
            if requireComma {
                try expect(type: .comma, error: "Expected comma between arguments")
                if typeof(.closeParen) {
                    break
                }
            }
            if typeof(.multiplicativeBinaryOperator), tokens[current].value == "*" {
                current += 1  // Consume *
                if dynArgs != nil || dynKwargs != nil {
                    throw JinjaError.syntax("Multiple dynamic positional arguments are not allowed.")
                }
                dynArgs = try parseExpression()
            } else if typeof(.multiplicativeBinaryOperator), tokens[current].value == "**" {
                current += 1  // Consume **
                if dynKwargs != nil {
                    throw JinjaError.syntax("Multiple dynamic keyword arguments are not allowed.")
                }
                dynKwargs = try parseExpression()
            } else {
                if typeof(.identifier), tokens.count > current + 1, tokens[current + 1].type == .equals {
                    // Parse keyword argument
                    guard let key = try parsePrimaryExpression() as? Identifier else {
                        throw JinjaError.syntax("Expected identifier for keyword argument key")
                    }
                    try expect(type: .equals, error: "Expected '=' after keyword argument key")
                    let value = try parseExpression()
                    if dynKwargs != nil {
                        throw JinjaError.syntax("Keyword arguments must be after dynamic keyword arguments")
                    }
                    kwargs.append(KeywordArgumentExpression(key: key, value: value))
                } else {
                    // Parse positional argument
                    if !kwargs.isEmpty || dynKwargs != nil {
                        throw JinjaError.syntax("Positional argument after keyword argument")
                    }
                    if dynArgs != nil {
                        throw JinjaError.syntax("Positional arguments must be after dynamic positional arguments")
                    }
                    args.append(try parseExpression())
                }
            }
            requireComma = true
        }
        try expect(type: .closeParen, error: "Expected closing parenthesis for arguments list")
        return (args, kwargs, dynArgs, dynKwargs)
    }

    func parseTestExpression() throws -> Expression {
        var operand = try parseFilterExpression()
        while typeof(.is) {
            current += 1
            let negate = typeof(.not)
            if negate {
                current += 1
            }
            var filter = try parsePrimaryExpression()
            if let boolLiteralFilter = filter as? BoolLiteral {
                filter = Identifier(value: String(boolLiteralFilter.value))
            } else if filter is NullLiteral {
                filter = Identifier(value: "none")
            }
            if let test = filter as? Identifier {
                operand = TestExpression(operand: operand, negate: negate, test: test)
            } else {
                throw JinjaError.syntax("Expected identifier for the test")
            }
        }
        return operand
    }

    func parseMultiplicativeExpression() throws -> Expression {
        var left = try parseTestExpression()
        while typeof(.multiplicativeBinaryOperator) {
            let operation = tokens[current]
            current += 1
            let right = try parseTestExpression()
            left = BinaryExpression(operation: operation, left: left, right: right)
        }
        return left
    }

    func parseAdditiveExpression() throws -> Expression {
        var left = try parseMultiplicativeExpression()
        while typeof(.additiveBinaryOperator) {
            let operation = tokens[current]
            current += 1
            let right = try parseMultiplicativeExpression()
            left = BinaryExpression(operation: operation, left: left, right: right)
        }
        return left
    }

    func parseComparisonExpression() throws -> Expression {
        var left = try parseAdditiveExpression()
        while typeof(.comparisonBinaryOperator) || typeof(.in) || typeof(.notIn)
            || (typeof(.is)
                && (tokens.count > current + 1
                    && (tokens[current + 1].type == .identifier || tokens[current + 1].type == .not)))
        {
            let operation = tokens[current]
            current += 1
            if operation.type == .is {
                if typeof(.not) {
                    current += 1
                    if typeof(.identifier), tokens[current].value == "none" {
                        current += 1
                        left = TestExpression(operand: left, negate: true, test: Identifier(value: "none"))
                        continue
                    } else {
                        throw JinjaError.syntax("Expected 'none' after 'is not'")
                    }
                } else if typeof(.identifier), tokens[current].value == "defined" {
                    current += 1
                    left = TestExpression(operand: left, negate: false, test: Identifier(value: "defined"))
                    continue
                } else {
                    throw JinjaError.syntax("Expected 'defined' or 'not' after 'is'")
                }
            } else if operation.type == .notIn {
                let right = try parseAdditiveExpression()
                left = BinaryExpression(operation: operation, left: left, right: right)
            } else {
                let right = try parseAdditiveExpression()
                left = BinaryExpression(operation: operation, left: left, right: right)
            }
        }
        return left
    }

    func parseLogicalNegationExpression() throws -> Expression {
        if typeof(.not) {
            let operation = tokens[current]
            current += 1
            let argument = try parseLogicalNegationExpression()
            return UnaryExpression(operation: operation, argument: argument)
        } else {
            return try parseComparisonExpression()
        }
    }

    func parseLogicalAndExpression() throws -> Expression {
        var left = try parseLogicalNegationExpression()
        while typeof(.and) {
            let operation = tokens[current]
            current += 1
            let right = try parseLogicalNegationExpression()
            left = BinaryExpression(operation: operation, left: left, right: right)
        }
        return left
    }

    func parseLogicalOrExpression() throws -> Expression {
        var left = try parseLogicalAndExpression()
        while typeof(.or) {
            current += 1  // Consume 'or'
            let right = try parseLogicalAndExpression()
            left = BinaryExpression(operation: Token(value: "or", type: .or), left: left, right: right)
        }
        return left
    }

    func parseTernaryExpression() throws -> Expression {
        let a = try parseLogicalOrExpression()
        if typeof(.if) {
            current += 1  // consume if token
            let predicate = try parseLogicalOrExpression()
            if typeof(.else) {
                // Ternary expression with else
                current += 1  // consume else token
                let b = try parseLogicalOrExpression()
                return If(test: predicate, body: [a], alternate: [b])
            } else {
                // Select expression on iterable
                return SelectExpression(iterable: a, test: predicate)
            }
        }
        return a
    }

    func parseExpression() throws -> Expression {
        try parseTernaryExpression()
    }

    func typeof(_ types: TokenType...) -> Bool {
        guard current + types.count <= tokens.count else {
            return false
        }
        for (index, type) in types.enumerated() {
            if type != tokens[current + index].type {
                return false
            }
        }
        return true
    }

    func parseSetStatement() throws -> Statement {
        let left = try parseExpression()
        if typeof(.equals) {
            current += 1  // consume equals
            // Parse the right-hand side as an expression
            let value = try parseExpression()
            try expect(type: .closeStatement, error: "Expected closing statement token")
            return Set(assignee: left, value: value)
        }
        // If there's no equals sign, treat it as an expression statement
        try expect(type: .closeStatement, error: "Expected closing statement token")
        return left
    }

    func parseIfStatement() throws -> Statement {
        let test = try parseExpression()
        try expect(type: .closeStatement, error: "Expected closing statement token")
        var body: [Statement] = []
        var alternate: [Statement] = []
        while !(tokens[current].type == .openStatement
            && (tokens[current + 1].type == .elseIf || tokens[current + 1].type == .else
                || tokens[current + 1].type == .endIf))
        {
            body.append(try parseAny())
        }
        if tokens[current].type == .openStatement, tokens[current + 1].type != .endIf {
            current += 1
            if typeof(.elseIf) {
                try expect(type: .elseIf, error: "Expected elseif token")
                alternate.append(try parseIfStatement())
            } else {
                try expect(type: .else, error: "Expected else token")
                try expect(type: .closeStatement, error: "Expected closing statement token")

                while !(tokens[current].type == .openStatement && tokens[current + 1].type == .endIf) {
                    alternate.append(try parseAny())
                }
            }
        }
        return If(test: test, body: body, alternate: alternate)
    }

    func parsePrimaryExpression() throws -> Expression {
        let token = tokens[current]
        switch token.type {
        case .numericLiteral:
            current += 1
            if let intValue = Int(token.value) {
                return NumericLiteral(value: intValue)
            } else if let doubleValue = Double(token.value) {
                return NumericLiteral(value: doubleValue)
            } else {
                throw JinjaError.parser("Invalid numeric literal: \(token.value)")
            }
        case .stringLiteral:
            current += 1
            return StringLiteral(value: token.value)
        case .booleanLiteral:
            current += 1
            return BoolLiteral(value: token.value == "true")
        case .nullLiteral:
            current += 1
            return NullLiteral()
        case .identifier:
            current += 1
            return Identifier(value: token.value)
        case .openParen:
            current += 1
            let expression = try parseExpressionSequence()
            if tokens[current].type != .closeParen {
                throw JinjaError.syntax("Expected closing parenthesis, got \(tokens[current].type) instead")
            }
            current += 1
            return expression
        case .openSquareBracket:
            current += 1
            var values: [Expression] = []
            while !typeof(.closeSquareBracket) {
                try values.append(parseExpression())
                if typeof(.comma) {
                    current += 1
                }
            }
            current += 1
            return ArrayLiteral(value: values)
        case .openCurlyBracket:
            current += 1
            var values = OrderedDictionary<String, Expression>()
            while !typeof(.closeCurlyBracket) {
                let key = try parseExpression()
                try expect(type: .colon, error: "Expected colon between key and value in object literal")
                let value = try parseExpression()

                if let key = key as? StringLiteral {
                    values[key.value] = value
                } else if let key = key as? Identifier {
                    values[key.value] = value
                } else {
                    throw JinjaError.syntax("Expected string literal or identifier as key in object literal")
                }

                if typeof(.comma) {
                    current += 1
                }
            }
            current += 1
            return ObjectLiteral(value: values)
        default:
            throw JinjaError.syntax("Unexpected token: \(token.type)")
        }
    }

    func parseExpressionSequence(primary: Bool = false) throws -> Expression {
        let fn = primary ? parsePrimaryExpression : parseExpression
        var expressions: [Expression] = try [fn()]
        let isTuple = typeof(.comma)
        while isTuple {
            current += 1  // consume comma
            try expressions.append(fn())
            if !typeof(.comma) {
                break
            }
        }
        // Return either a tuple or single expression
        return isTuple ? TupleLiteral(value: expressions) : expressions[0]
    }

    func not(_ types: TokenType...) -> Bool {
        guard current + types.count <= tokens.count else {
            return false
        }
        return types.enumerated().contains { i, type -> Bool in
            type != tokens[current + i].type
        }
    }

    func parseForStatement() throws -> Statement {
        let loopVariable = try parseExpressionSequence(primary: true)
        if !(loopVariable is Identifier || loopVariable is TupleLiteral) {
            throw JinjaError.syntax(
                "Expected identifier/tuple for the loop variable, got \(type(of: loopVariable)) instead"
            )
        }
        try expect(type: .in, error: "Expected `in` keyword following loop variable")
        let iterable = try parseExpression()
        // Handle optional if condition for filtering
        var test: Expression? = nil
        if typeof(.if) {
            current += 1  // consume if token
            test = try parseExpression()
        }
        try expect(type: .closeStatement, error: "Expected closing statement token")
        var body: [Statement] = []
        var defaultBlock: [Statement] = []
        while not(.openStatement, .endFor) && not(.openStatement, .else) {
            body.append(try parseAny())
        }
        if typeof(.openStatement, .else) {
            current += 1  // consume {%
            try expect(type: .else, error: "Expected else token")
            try expect(type: .closeStatement, error: "Expected closing statement token")

            while not(.openStatement, .endFor) {
                defaultBlock.append(try parseAny())
            }
        }
        return For(
            loopvar: loopVariable,
            iterable: iterable,
            body: body,
            defaultBlock: defaultBlock,
            test: test
        )
    }

    func parseMacroStatement() throws -> Macro {
        let name = try parsePrimaryExpression()
        if !(name is Identifier) {
            throw JinjaError.syntax("Expected identifier following macro statement")
        }
        let args = try parseArgs()
        try expect(type: .closeStatement, error: "Expected closing statement token")
        var body: [Statement] = []
        while not(.openStatement, .endMacro) {
            body.append(try parseAny())
        }
        return Macro(name: name as! Identifier, args: args, body: body)
    }

    func parseJinjaStatement() throws -> Statement {
        // Consume {% %} tokens
        try expect(type: .openStatement, error: "Expected opening statement token")
        var result: Statement

        switch tokens[current].type {
        case .set:
            current += 1  // consume 'set' token
            result = try parseSetStatement()
        case .if:
            current += 1  // consume 'if' token
            result = try parseIfStatement()
            try expect(type: .openStatement, error: "Expected {% token")
            try expect(type: .endIf, error: "Expected endif token")
            try expect(type: .closeStatement, error: "Expected %} token")
        case .macro:
            current += 1  // consume 'macro' token
            result = try parseMacroStatement()
            try expect(type: .openStatement, error: "Expected {% token")
            try expect(type: .endMacro, error: "Expected endmacro token")
            try expect(type: .closeStatement, error: "Expected %} token")
        case .for:
            current += 1  // consume 'for' token
            result = try parseForStatement()
            try expect(type: .openStatement, error: "Expected {% token")
            try expect(type: .endFor, error: "Expected endfor token")
            try expect(type: .closeStatement, error: "Expected %} token")
        default:
            // Handle expressions within statements
            result = try parseExpression()
            try expect(type: .closeStatement, error: "Expected closing statement token")
        }

        return result
    }

    func parseJinjaExpression() throws -> Statement {
        try expect(type: .openExpression, error: "Expected opening expression token")
        let result = try parseExpression()
        try expect(type: .closeExpression, error: "Expected closing expression token")
        return result
    }

    func parseAny() throws -> Statement {
        switch tokens[current].type {
        case .text:
            return try parseText()
        case .openStatement:
            return try parseJinjaStatement()
        case .openExpression:
            return try parseJinjaExpression()
        default:
            throw JinjaError.syntax("Unexpected token type: \(tokens[current].type)")
        }
    }

    while current < tokens.count {
        try program.body.append(parseAny())
    }

    return program
}
