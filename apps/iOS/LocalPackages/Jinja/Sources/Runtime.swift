//
//  Runtime.swift
//
//
//  Created by John Mai on 2024/3/22.
//

import Foundation
import OrderedCollections

protocol RuntimeValue {
    associatedtype ValueType

    var value: ValueType { get }
    var builtins: [String: any RuntimeValue] { get set }

    func bool() -> Bool
}

struct NumericValue: RuntimeValue {
    var value: any Numeric
    var builtins: [String: any RuntimeValue] = [:]

    func bool() -> Bool {
        if let intValue = self.value as? Int {
            return intValue != 0
        } else if let doubleValue = self.value as? Double {
            return doubleValue != 0.0
        }
        return false
    }
}

struct BooleanValue: RuntimeValue {
    var value: Bool
    var builtins: [String: any RuntimeValue] = [:]

    func bool() -> Bool {
        self.value
    }
}

struct NullValue: RuntimeValue {
    let value: Any? = nil
    var builtins: [String: any RuntimeValue] = [:]

    func bool() -> Bool {
        false
    }
}

struct UndefinedValue: RuntimeValue {
    let value: Any? = nil
    var builtins: [String: any RuntimeValue] = [:]

    func bool() -> Bool {
        false
    }
}

struct ArrayValue: RuntimeValue {
    var value: [any RuntimeValue]
    var builtins: [String: any RuntimeValue] = [:]

    init(value: [any RuntimeValue]) {
        self.value = value
        self.builtins["length"] = FunctionValue(value: { _, _ in
            NumericValue(value: value.count)
        })
    }

    func bool() -> Bool {
        return !self.value.isEmpty
    }
}

struct TupleValue: RuntimeValue {
    var value: [any RuntimeValue]
    var builtins: [String: any RuntimeValue] = [:]

    init(value: [any RuntimeValue]) {
        self.value = value
        self.builtins["length"] = FunctionValue(value: { _, _ in
            NumericValue(value: value.count)
        })
    }

    func bool() -> Bool {
        !self.value.isEmpty
    }
}

class ObjectValue: RuntimeValue, Sequence {
    var storage: OrderedDictionary<String, any RuntimeValue>
    var builtins: [String: any RuntimeValue]

    var value: [String: any RuntimeValue] { Dictionary(uniqueKeysWithValues: storage.map { ($0, $1) }) }
    var orderedKeys: [String] { Array(storage.keys) }

    init(value: [String: any RuntimeValue], keyOrder: [String]? = nil) {
        // If keyOrder is provided, use it; otherwise, maintain the original order from the dictionary
        let orderedKeys = keyOrder ?? Array(value.keys)
        let orderedPairs = orderedKeys.compactMap { key in
            value[key].map { (key, $0) }
        }

        // Recursively create OrderedDictionary for nested objects
        let processedPairs = orderedPairs.map { key, value -> (String, any RuntimeValue) in
            if let objectValue = value as? ObjectValue {
                // Already an ObjectValue, use it directly
                return (key, objectValue)
            } else if let dictValue = value.value as? [String: any RuntimeValue] {
                // If the value contains a dictionary, convert it to ObjectValue
                return (key, ObjectValue(value: dictValue))
            }
            return (key, value)
        }

        self.storage = OrderedDictionary(uniqueKeysWithValues: processedPairs)
        self.builtins = [
            "get": FunctionValue(value: { args, _ in
                guard let key = args[0] as? StringValue else {
                    throw JinjaError.runtime("Object key must be a string: got \(type(of: args[0]))")
                }
                if let value = value[key.value] {
                    return value
                } else if args.count > 1 {
                    return args[1]
                }
                return NullValue()
            }),
            "items": FunctionValue(value: { _, _ in
                ArrayValue(
                    value: orderedPairs.map { key, value in
                        ArrayValue(value: [StringValue(value: key), value])
                    }
                )
            }),
            "keys": FunctionValue(value: { _, _ in
                ArrayValue(value: orderedKeys.map { StringValue(value: $0) })
            }),
        ]
    }

    func setValue(key: String, value: any RuntimeValue) {
        storage[key] = value
    }

    func bool() -> Bool {
        !storage.isEmpty
    }

    func makeIterator() -> OrderedDictionary<String, any RuntimeValue>.Iterator {
        return storage.makeIterator()
    }
}

struct FunctionValue: RuntimeValue {
    var value: ([any RuntimeValue], Environment) throws -> any RuntimeValue
    var builtins: [String: any RuntimeValue] = [:]

    func bool() -> Bool {
        true
    }
}

struct StringValue: RuntimeValue {
    var value: String
    var builtins: [String: any RuntimeValue] = [:]

    init(value: String) {
        self.value = value
        self.builtins = [
            "upper": FunctionValue(value: { _, _ in
                StringValue(value: value.uppercased())
            }),
            "lower": FunctionValue(value: { _, _ in
                StringValue(value: value.lowercased())
            }),
            "strip": FunctionValue(value: { _, _ in
                StringValue(value: value.trimmingCharacters(in: .whitespacesAndNewlines))
            }),
            "title": FunctionValue(value: { _, _ in
                StringValue(value: value.titleCase())
            }),
            "length": FunctionValue(value: { _, _ in
                NumericValue(value: value.count)
            }),
            "rstrip": FunctionValue(value: { _, _ in
                StringValue(value: value.replacingOccurrences(of: "\\s+$", with: "", options: .regularExpression))
            }),
            "lstrip": FunctionValue(value: { _, _ in
                StringValue(value: value.replacingOccurrences(of: "^\\s+", with: "", options: .regularExpression))
            }),
            "split": FunctionValue(value: { args, _ in
                guard let separatorArg = args.first as? StringValue else {
                    // Default split by whitespace if no separator is provided or if it's not a string
                    // (This mimics Python's str.split() behavior loosely)
                    let components = value.split(whereSeparator: { $0.isWhitespace })
                    return ArrayValue(value: components.map { StringValue(value: String($0)) })
                }
                let separator = separatorArg.value
                // TODO: Add optional maxsplit argument handling if needed
                let components = value.components(separatedBy: separator)
                return ArrayValue(value: components.map { StringValue(value: $0) })
            }),
            "startswith": FunctionValue(value: { args, _ in
                guard let prefixArg = args.first as? StringValue else {
                    throw JinjaError.runtime("startswith requires a string prefix argument")
                }
                return BooleanValue(value: value.hasPrefix(prefixArg.value))
            }),
            "endswith": FunctionValue(value: { args, _ in
                guard let suffixArg = args.first as? StringValue else {
                    throw JinjaError.runtime("endswith requires a string suffix argument")
                }
                return BooleanValue(value: value.hasSuffix(suffixArg.value))
            }),
            "replace": FunctionValue(value: { args, _ in
                guard args.count >= 2 else {
                    throw JinjaError.runtime("replace() requires at least two arguments")
                }

                guard let oldValue = args[0] as? StringValue else {
                    throw JinjaError.runtime("replace() first argument must be a string")
                }

                guard let newValue = args[1] as? StringValue else {
                    throw JinjaError.runtime("replace() second argument must be a string")
                }

                var count: any RuntimeValue = NullValue()
                if args.count > 2 {
                    if let countValue = args[2] as? KeywordArgumentsValue {
                        return countValue.value["count"] ?? NullValue()
                    } else {
                        count = args[2]
                    }
                }

                if !(count is NumericValue || count is NullValue) {
                    throw JinjaError.runtime("replace() count argument must be a number or null")
                }

                if let countValue = count as? NumericValue, let maxReplacements = countValue.value as? Int {
                    return StringValue(
                        value: value.replacingOccurrences(
                            of: oldValue.value,
                            with: newValue.value,
                            count: maxReplacements
                        )
                    )
                } else {
                    return StringValue(value: value.replacingOccurrences(of: oldValue.value, with: newValue.value))
                }
            }),
        ]
    }

    func bool() -> Bool {
        !self.value.isEmpty
    }
}

struct Interpreter {
    var global: Environment

    init(env: Environment?) {
        self.global = env ?? Environment()
    }

    func run(program: Program) throws -> any RuntimeValue {
        try self.evaluate(statement: program, environment: self.global)
    }

    func evaluateBlock(statements: [Statement], environment: Environment) throws -> StringValue {
        var result = ""
        for statement in statements {
            let lastEvaluated = try self.evaluate(statement: statement, environment: environment)
            if !(lastEvaluated is NullValue), !(lastEvaluated is UndefinedValue) {
                if let stringValue = lastEvaluated as? StringValue {
                    result += stringValue.value
                } else if let numericValue = lastEvaluated as? NumericValue {
                    result += String(describing: numericValue.value)
                } else if let booleanValue = lastEvaluated as? BooleanValue {
                    result += String(booleanValue.value)
                } else if let arrayValue = lastEvaluated as? ArrayValue {
                    // Convert array to JSON string
                    result += try toJSON(arrayValue)
                } else if let objectValue = lastEvaluated as? ObjectValue {
                    // Convert object to JSON string
                    result += try toJSON(objectValue)
                } else {
                    throw JinjaError.runtime("Cannot convert to string: \(type(of: lastEvaluated))")
                }
            }
        }
        return StringValue(value: result)
    }

    func evalProgram(program: Program, environment: Environment) throws -> StringValue {
        try self.evaluateBlock(statements: program.body, environment: environment)
    }

    func evaluateSet(node: Set, environment: Environment) throws -> NullValue {
        let rhs = try self.evaluate(statement: node.value, environment: environment)
        if let identifier = node.assignee as? Identifier {
            let variableName = identifier.value
            try environment.setVariable(name: variableName, value: rhs)
        } else if let member = node.assignee as? MemberExpression {
            let object = try self.evaluate(statement: member.object, environment: environment)
            guard let objectValue = object as? ObjectValue else {
                throw JinjaError.runtime("Cannot assign to member of non-object")
            }
            guard let property = member.property as? Identifier else {
                throw JinjaError.runtime("Cannot assign to member with non-identifier property")
            }
            // Modify the copy
            objectValue.setValue(key: property.value, value: rhs)
            // Update the environment with the modified copy
            if let parentIdentifier = member.object as? Identifier {
                try environment.setVariable(name: parentIdentifier.value, value: objectValue)
            } else {
                throw JinjaError.runtime("Cannot assign to computed member expression")
            }
        } else {
            throw JinjaError.runtime("Invalid LHS inside assignment expression: \(node.assignee)")
        }
        return NullValue()
    }

    func evaluateIf(node: If, environment: Environment) throws -> StringValue {
        // Special handling for direct variable checks
        if let identifier = node.test as? Identifier {
            // For cases like {% if thinking %}, get the variable directly
            let value = environment.lookupVariable(name: identifier.value)
            // Use the bool method which will return false for undefined values
            let testResult = value.bool()
            return try self.evaluateBlock(statements: testResult ? node.body : node.alternate, environment: environment)
        }

        // For non-identifier checks, evaluate normally
        let test = try self.evaluate(statement: node.test, environment: environment)
        return try self.evaluateBlock(statements: test.bool() ? node.body : node.alternate, environment: environment)
    }

    func evaluateIdentifier(node: Identifier, environment: Environment) throws -> any RuntimeValue {
        let value = environment.lookupVariable(name: node.value)
        return value
    }

    func evaluateFor(node: For, environment: Environment) throws -> StringValue {
        // Scope for the for loop
        let scope = Environment(parent: environment)
        let test: Expression?
        let iterable: any RuntimeValue
        if let selectExpression = node.iterable as? SelectExpression {
            iterable = try self.evaluate(statement: selectExpression.iterable, environment: scope)
            test = selectExpression.test
        } else {
            iterable = try self.evaluate(statement: node.iterable, environment: scope)
            test = nil
        }
        var items: [any RuntimeValue] = []
        var scopeUpdateFunctions: [(Environment) throws -> Void] = []
        // Keep track of the indices of the original iterable that passed the test
        var filteredIndices: [Int] = []
        var originalIndex = 0
        // Handle ArrayValue
        if let arrayIterable = iterable as? ArrayValue {
            for current in arrayIterable.value {
                let loopScope = Environment(parent: scope)
                var scopeUpdateFunction: (Environment) throws -> Void
                if let identifier = node.loopvar as? Identifier {
                    scopeUpdateFunction = { scope in
                        try scope.setVariable(name: identifier.value, value: current)
                    }
                } else if let tupleLiteral = node.loopvar as? TupleLiteral {
                    guard let currentArray = current as? ArrayValue else {
                        throw JinjaError.runtime("Cannot unpack non-iterable type: \(type(of: current))")
                    }
                    if tupleLiteral.value.count != currentArray.value.count {
                        throw JinjaError.runtime(
                            "Too \(tupleLiteral.value.count > currentArray.value.count ? "few" : "many") items to unpack"
                        )
                    }
                    scopeUpdateFunction = { scope in
                        for (i, value) in tupleLiteral.value.enumerated() {
                            guard let identifier = value as? Identifier else {
                                throw JinjaError.runtime("Cannot unpack non-identifier type: \(type(of: value))")
                            }
                            try scope.setVariable(name: identifier.value, value: currentArray.value[i])
                        }
                    }
                } else {
                    throw JinjaError.runtime("Invalid loop variable(s): \(type(of: node.loopvar))")
                }
                // Evaluate the test before adding the item
                if let test {
                    try scopeUpdateFunction(loopScope)
                    let testValue = try self.evaluate(statement: test, environment: loopScope)
                    if !testValue.bool() {
                        originalIndex += 1
                        continue
                    }
                }
                items.append(current)
                scopeUpdateFunctions.append(scopeUpdateFunction)
                filteredIndices.append(originalIndex)
                originalIndex += 1
            }
            // Handle StringValue as a special case
        } else if let stringIterable = iterable as? StringValue {
            // Treat the string as an iterable of characters
            for char in stringIterable.value {
                let current = StringValue(value: String(char))
                let loopScope = Environment(parent: scope)
                var scopeUpdateFunction: (Environment) throws -> Void
                if let identifier = node.loopvar as? Identifier {
                    scopeUpdateFunction = { scope in
                        try scope.setVariable(name: identifier.value, value: current)
                    }
                } else {
                    throw JinjaError.runtime("Invalid loop variable(s): \(type(of: node.loopvar))")
                }
                // Evaluate the test before adding the item
                if let test = test {
                    try scopeUpdateFunction(loopScope)
                    let testValue = try self.evaluate(statement: test, environment: loopScope)
                    if !testValue.bool() {
                        originalIndex += 1
                        continue
                    }
                }
                items.append(current)
                scopeUpdateFunctions.append(scopeUpdateFunction)
                filteredIndices.append(originalIndex)
                originalIndex += 1
            }
            // Handle ObjectValue (dictionary)
        } else if let objectIterable = iterable as? ObjectValue {
            // Treat the dictionary as an iterable of key-value pairs
            for (key, value) in objectIterable {
                let current = ArrayValue(value: [StringValue(value: key), value])
                let loopScope = Environment(parent: scope)
                var scopeUpdateFunction: (Environment) throws -> Void
                if let identifier = node.loopvar as? Identifier {
                    scopeUpdateFunction = { scope in
                        try scope.setVariable(name: identifier.value, value: current)
                    }
                } else if let tupleLiteral = node.loopvar as? TupleLiteral {
                    // Support unpacking of key-value pairs into two variables
                    if tupleLiteral.value.count != 2 {
                        throw JinjaError.runtime(
                            "Cannot unpack dictionary entry: expected 2 variables, got \(tupleLiteral.value.count)"
                        )
                    }
                    guard let keyIdentifier = tupleLiteral.value[0] as? Identifier else {
                        throw JinjaError.runtime(
                            "Cannot unpack dictionary entry into non-identifier: \(type(of: tupleLiteral.value[0]))"
                        )
                    }
                    guard let valueIdentifier = tupleLiteral.value[1] as? Identifier else {
                        throw JinjaError.runtime(
                            "Cannot unpack dictionary entry into non-identifier: \(type(of: tupleLiteral.value[1]))"
                        )
                    }
                    scopeUpdateFunction = { scope in
                        try scope.setVariable(name: keyIdentifier.value, value: StringValue(value: key))
                        try scope.setVariable(name: valueIdentifier.value, value: value)
                    }
                } else {
                    throw JinjaError.runtime("Invalid loop variable(s): \(type(of: node.loopvar))")
                }
                // Evaluate the test before adding the item
                if let test = test {
                    try scopeUpdateFunction(loopScope)
                    let testValue = try self.evaluate(statement: test, environment: loopScope)
                    if !testValue.bool() {
                        originalIndex += 1
                        continue
                    }
                }
                items.append(current)
                scopeUpdateFunctions.append(scopeUpdateFunction)
                filteredIndices.append(originalIndex)
                originalIndex += 1
            }
        } else {
            throw JinjaError.runtime("Expected iterable type in for loop: got \(type(of: iterable))")
        }
        var result = ""
        var noIteration = true
        for i in 0 ..< items.count {
            // Get the previous and next items that passed the filter
            let previousIndex = filteredIndices.firstIndex(of: filteredIndices[i])! - 1
            let nextIndex = filteredIndices.firstIndex(of: filteredIndices[i])! + 1
            let previtem: any RuntimeValue
            if previousIndex >= 0 {
                let previousFilteredIndex = filteredIndices[previousIndex]
                if let arrayIterable = iterable as? ArrayValue {
                    previtem = arrayIterable.value[previousFilteredIndex]
                } else if let stringIterable = iterable as? StringValue {
                    let index = stringIterable.value.index(
                        stringIterable.value.startIndex,
                        offsetBy: previousFilteredIndex
                    )
                    previtem = StringValue(value: String(stringIterable.value[index]))
                } else if let objectIterable = iterable as? ObjectValue {
                    let (key, value) = objectIterable.storage.elements[previousFilteredIndex]
                    previtem = ArrayValue(value: [StringValue(value: key), value])
                } else {
                    previtem = UndefinedValue()
                }
            } else {
                previtem = UndefinedValue()
            }
            let nextitem: any RuntimeValue
            if nextIndex < filteredIndices.count {
                let nextFilteredIndex = filteredIndices[nextIndex]
                if let arrayIterable = iterable as? ArrayValue {
                    nextitem = arrayIterable.value[nextFilteredIndex]
                } else if let stringIterable = iterable as? StringValue {
                    let index = stringIterable.value.index(stringIterable.value.startIndex, offsetBy: nextFilteredIndex)
                    nextitem = StringValue(value: String(stringIterable.value[index]))
                } else if let objectIterable = iterable as? ObjectValue {
                    let (key, value) = objectIterable.storage.elements[nextFilteredIndex]
                    nextitem = ArrayValue(value: [StringValue(value: key), value])
                } else {
                    nextitem = UndefinedValue()
                }
            } else {
                nextitem = UndefinedValue()
            }
            let loop: [String: any RuntimeValue] = [
                "index": NumericValue(value: i + 1),
                "index0": NumericValue(value: i),
                "revindex": NumericValue(value: items.count - i),
                "revindex0": NumericValue(value: items.count - i - 1),
                "first": BooleanValue(value: i == 0),
                "last": BooleanValue(value: i == items.count - 1),
                "length": NumericValue(value: items.count),
                "previtem": previtem,
                "nextitem": nextitem,
            ]
            try scope.setVariable(name: "loop", value: ObjectValue(value: loop))
            try scopeUpdateFunctions[i](scope)
            let evaluated = try self.evaluateBlock(statements: node.body, environment: scope)
            result += evaluated.value
            noIteration = false
        }
        if noIteration {
            let defaultEvaluated = try self.evaluateBlock(statements: node.defaultBlock, environment: scope)
            result += defaultEvaluated.value
        }
        return StringValue(value: result)
    }

    func evaluateBinaryExpression(node: BinaryExpression, environment: Environment) throws -> any RuntimeValue {
        let left = try self.evaluate(statement: node.left, environment: environment)
        let right = try self.evaluate(statement: node.right, environment: environment)
        // Handle 'or'
        if node.operation.value == "or" {
            if left.bool() {
                return left
            } else {
                return right
            }
        }
        // Handle 'and'
        if node.operation.value == "and" {
            if !left.bool() {
                return left
            } else {
                return right
            }
        }
        // ==
        if node.operation.value == "==" {
            // Handle array indexing for right operand
            if let memberExpr = node.right as? MemberExpression,
                let arrayValue = try self.evaluate(statement: memberExpr.object, environment: environment)
                    as? ArrayValue,
                let indexExpr = memberExpr.property as? NumericLiteral,
                let index = indexExpr.value as? Int
            {

                // Handle negative indices
                let actualIndex = index < 0 ? arrayValue.value.count + index : index
                if actualIndex >= 0 && actualIndex < arrayValue.value.count {
                    let rightValue = arrayValue.value[actualIndex]
                    return BooleanValue(value: try areEqual(left, rightValue))
                }
            }

            return BooleanValue(value: try areEqual(left, right))
        }
        // !=
        if node.operation.value == "!=" {
            if let left = left as? StringValue, let right = right as? StringValue {
                return BooleanValue(value: left.value != right.value)
            } else if let left = left as? NumericValue, let right = right as? NumericValue {
                if let leftInt = left.value as? Int, let rightInt = right.value as? Int {
                    return BooleanValue(value: leftInt != rightInt)
                } else if let leftDouble = left.value as? Double, let rightDouble = right.value as? Double {
                    return BooleanValue(value: leftDouble != rightDouble)
                } else if let leftInt = left.value as? Int, let rightDouble = right.value as? Double {
                    return BooleanValue(value: Double(leftInt) != rightDouble)
                } else if let leftDouble = left.value as? Double, let rightInt = right.value as? Int {
                    return BooleanValue(value: leftDouble != Double(rightInt))
                } else {
                    throw JinjaError.runtime("Unsupported numeric types for inequality comparison")
                }
            } else if let left = left as? BooleanValue, let right = right as? BooleanValue {
                return BooleanValue(value: left.value != right.value)
            } else if left is NullValue, right is NullValue {
                return BooleanValue(value: false)
            } else if left is UndefinedValue, right is UndefinedValue {
                return BooleanValue(value: false)
            } else if type(of: left) == type(of: right) {
                return BooleanValue(value: true)
            } else {
                return BooleanValue(value: true)
            }
        }

        // Handle operations with undefined or null values
        if left is UndefinedValue || right is UndefinedValue || left is NullValue || right is NullValue {
            // Boolean operations return false
            if ["and", "or", "==", "!=", ">", "<", ">=", "<=", "in", "not in"].contains(node.operation.value) {
                return BooleanValue(value: false)
            }

            // String concatenation with undefined/null
            if node.operation.value == "+" {
                if left is StringValue && !(right is UndefinedValue || right is NullValue) {
                    return left
                } else if right is StringValue && !(left is UndefinedValue || left is NullValue) {
                    return right
                }
                return StringValue(value: "")
            }

            // Math operations with undefined/null
            if ["-", "*", "/", "%"].contains(node.operation.value) {
                return NumericValue(value: 0)
            }

            return BooleanValue(value: false)
        } else if (node.operation.value == "~") {
            return StringValue(value: "\(left.value)\(right.value)")
        } else if let left = left as? NumericValue, let right = right as? NumericValue {
            switch node.operation.value {
            case "+":
                if let leftInt = left.value as? Int, let rightInt = right.value as? Int {
                    return NumericValue(value: leftInt + rightInt)
                } else if let leftDouble = left.value as? Double, let rightDouble = right.value as? Double {
                    return NumericValue(value: leftDouble + rightDouble)
                } else if let leftInt = left.value as? Int, let rightDouble = right.value as? Double {
                    return NumericValue(value: Double(leftInt) + rightDouble)
                } else if let leftDouble = left.value as? Double, let rightInt = right.value as? Int {
                    return NumericValue(value: leftDouble + Double(rightInt))
                } else {
                    throw JinjaError.runtime("Unsupported numeric types for addition")
                }
            case "-":
                if let leftInt = left.value as? Int, let rightInt = right.value as? Int {
                    return NumericValue(value: leftInt - rightInt)
                } else if let leftDouble = left.value as? Double, let rightDouble = right.value as? Double {
                    return NumericValue(value: leftDouble - rightDouble)
                } else if let leftInt = left.value as? Int, let rightDouble = right.value as? Double {
                    return NumericValue(value: Double(leftInt) - rightDouble)
                } else if let leftDouble = left.value as? Double, let rightInt = right.value as? Int {
                    return NumericValue(value: leftDouble - Double(rightInt))
                } else {
                    throw JinjaError.runtime("Unsupported numeric types for subtraction")
                }
            case "*":
                if let leftInt = left.value as? Int, let rightInt = right.value as? Int {
                    return NumericValue(value: leftInt * rightInt)
                } else if let leftDouble = left.value as? Double, let rightDouble = right.value as? Double {
                    return NumericValue(value: leftDouble * rightDouble)
                } else if let leftInt = left.value as? Int, let rightDouble = right.value as? Double {
                    return NumericValue(value: Double(leftInt) * rightDouble)
                } else if let leftDouble = left.value as? Double, let rightInt = right.value as? Int {
                    return NumericValue(value: leftDouble * Double(rightInt))
                } else {
                    throw JinjaError.runtime("Unsupported numeric types for multiplication")
                }
            case "/":
                if let leftInt = left.value as? Int, let rightInt = right.value as? Int {
                    return NumericValue(value: leftInt / rightInt)
                } else if let leftDouble = left.value as? Double, let rightDouble = right.value as? Double {
                    return NumericValue(value: leftDouble / rightDouble)
                } else if let leftInt = left.value as? Int, let rightDouble = right.value as? Double {
                    return NumericValue(value: Double(leftInt) / rightDouble)
                } else if let leftDouble = left.value as? Double, let rightInt = right.value as? Int {
                    return NumericValue(value: leftDouble / Double(rightInt))
                } else {
                    throw JinjaError.runtime("Unsupported numeric types for division")
                }
            case "%":
                if let leftInt = left.value as? Int, let rightInt = right.value as? Int {
                    return NumericValue(value: leftInt % rightInt)
                } else {
                    throw JinjaError.runtime("Unsupported numeric types for modulus")
                }
            case "<":
                if let leftInt = left.value as? Int, let rightInt = right.value as? Int {
                    return BooleanValue(value: leftInt < rightInt)
                } else if let leftDouble = left.value as? Double, let rightDouble = right.value as? Double {
                    return BooleanValue(value: leftDouble < rightDouble)
                } else if let leftInt = left.value as? Int, let rightDouble = right.value as? Double {
                    return BooleanValue(value: Double(leftInt) < rightDouble)
                } else if let leftDouble = left.value as? Double, let rightInt = right.value as? Int {
                    return BooleanValue(value: leftDouble < Double(rightInt))
                } else {
                    throw JinjaError.runtime("Unsupported numeric types for less than comparison")
                }
            case ">":
                if let leftInt = left.value as? Int, let rightInt = right.value as? Int {
                    return BooleanValue(value: leftInt > rightInt)
                } else if let leftDouble = left.value as? Double, let rightDouble = right.value as? Double {
                    return BooleanValue(value: leftDouble > rightDouble)
                } else if let leftInt = left.value as? Int, let rightDouble = right.value as? Double {
                    return BooleanValue(value: Double(leftInt) > rightDouble)
                } else if let leftDouble = left.value as? Double, let rightInt = right.value as? Int {
                    return BooleanValue(value: leftDouble > Double(rightInt))
                } else {
                    throw JinjaError.runtime("Unsupported numeric types for greater than comparison")
                }
            case ">=":
                if let leftInt = left.value as? Int, let rightInt = right.value as? Int {
                    return BooleanValue(value: leftInt >= rightInt)
                } else if let leftDouble = left.value as? Double, let rightDouble = right.value as? Double {
                    return BooleanValue(value: leftDouble >= rightDouble)
                } else if let leftInt = left.value as? Int, let rightDouble = right.value as? Double {
                    return BooleanValue(value: Double(leftInt) >= rightDouble)
                } else if let leftDouble = left.value as? Double, let rightInt = right.value as? Int {
                    return BooleanValue(value: leftDouble >= Double(rightInt))
                } else {
                    throw JinjaError.runtime("Unsupported numeric types for greater than or equal to comparison")
                }
            case "<=":
                if let leftInt = left.value as? Int, let rightInt = right.value as? Int {
                    return BooleanValue(value: leftInt <= rightInt)
                } else if let leftDouble = left.value as? Double, let rightDouble = right.value as? Double {
                    return BooleanValue(value: leftDouble <= rightDouble)
                } else if let leftInt = left.value as? Int, let rightDouble = right.value as? Double {
                    return BooleanValue(value: Double(leftInt) <= rightDouble)
                } else if let leftDouble = left.value as? Double, let rightInt = right.value as? Int {
                    return BooleanValue(value: leftDouble <= Double(rightInt))
                } else {
                    throw JinjaError.runtime("Unsupported numeric types for less than or equal to comparison")
                }
            default:
                throw JinjaError.runtime("Unknown operation type:\(node.operation.value)")
            }
        } else if let left = left as? ArrayValue, let right = right as? ArrayValue {
            switch node.operation.value {
            case "+":
                return ArrayValue(value: left.value + right.value)
            default:
                throw JinjaError.runtime("Unknown operation type:\(node.operation.value)")
            }
        } else if let right = right as? ArrayValue {
            let member: Bool
            if let left = left as? StringValue {
                member = right.value.contains {
                    if let item = $0 as? StringValue {
                        return item.value == left.value
                    }
                    return false
                }
            } else if let left = left as? NumericValue {
                member = right.value.contains {
                    if let item = $0 as? NumericValue {
                        return item.value as! Int == left.value as! Int
                    }
                    return false
                }
            } else if let left = left as? BooleanValue {
                member = right.value.contains {
                    if let item = $0 as? BooleanValue {
                        return item.value == left.value
                    }
                    return false
                }
            } else {
                throw JinjaError.runtime("Unsupported left type for 'in'/'not in' operation with ArrayValue")
            }
            switch node.operation.value {
            case "in":
                return BooleanValue(value: member)
            case "not in":
                return BooleanValue(value: !member)
            default:
                throw JinjaError.runtime("Unknown operation type:\(node.operation.value)")
            }
        }
        if let left = left as? StringValue {
            switch node.operation.value {
            case "+":
                let rightValue: String
                if let rightString = right as? StringValue {
                    rightValue = rightString.value
                } else if let rightNumeric = right as? NumericValue {
                    rightValue = String(describing: rightNumeric.value)
                } else if let rightBoolean = right as? BooleanValue {
                    rightValue = String(rightBoolean.value)
                } else if right is UndefinedValue || right is NullValue {
                    rightValue = ""
                } else {
                    throw JinjaError.runtime("Unsupported right operand type for string concatenation")
                }
                return StringValue(value: left.value + rightValue)
            case "in":
                if let right = right as? StringValue {
                    return BooleanValue(value: right.value.contains(left.value))
                } else if let right = right as? ObjectValue {
                    return BooleanValue(value: right.value.keys.contains(left.value))
                } else if let right = right as? ArrayValue {
                    return BooleanValue(
                        value: right.value.contains {
                            if let item = $0 as? StringValue {
                                return item.value == left.value
                            }
                            return false
                        }
                    )
                } else {
                    throw JinjaError.runtime("Right operand of 'in' must be a StringValue, ArrayValue, or ObjectValue")
                }
            case "not in":
                if let right = right as? StringValue {
                    return BooleanValue(value: !right.value.contains(left.value))
                } else if let right = right as? ObjectValue {
                    return BooleanValue(value: !right.value.keys.contains(left.value))
                } else if let right = right as? ArrayValue {
                    return BooleanValue(
                        value: !right.value.contains {
                            if let item = $0 as? StringValue {
                                return item.value == left.value
                            }
                            return false
                        }
                    )
                } else {
                    throw JinjaError.runtime(
                        "Right operand of 'not in' must be a StringValue, ArrayValue, or ObjectValue"
                    )
                }
            default:
                break
            }
        } else if let right = right as? StringValue {
            if node.operation.value == "+" {
                if let leftString = left as? StringValue {
                    return StringValue(value: leftString.value + right.value)
                } else if let leftNumeric = left as? NumericValue {
                    return StringValue(value: String(describing: leftNumeric.value) + right.value)
                } else if let leftBoolean = left as? BooleanValue {
                    return StringValue(value: String(leftBoolean.value) + right.value)
                } else {
                    throw JinjaError.runtime("Unsupported left operand type for string concatenation")
                }
            }
        }
        if let left = left as? StringValue, let right = right as? ObjectValue {
            switch node.operation.value {
            case "in":
                return BooleanValue(value: right.value.keys.contains(left.value))
            case "not in":
                return BooleanValue(value: !right.value.keys.contains(left.value))
            default:
                throw JinjaError.runtime(
                    "Unsupported operation '\(node.operation.value)' between StringValue and ObjectValue"
                )
            }
        }
        throw JinjaError.syntax(
            "Unknown operator '\(node.operation.value)' between \(type(of:left)) and \(type(of:right))"
        )
    }

    func evaluateSliceExpression(
        object: any RuntimeValue,
        expr: SliceExpression,
        environment: Environment
    ) throws -> any RuntimeValue {
        if !(object is ArrayValue || object is StringValue) {
            throw JinjaError.runtime("Slice object must be an array or string")
        }
        let start = try self.evaluate(statement: expr.start, environment: environment)
        let stop = try self.evaluate(statement: expr.stop, environment: environment)
        let step = try self.evaluate(statement: expr.step, environment: environment)
        if !(start is NumericValue || start is UndefinedValue) {
            throw JinjaError.runtime("Slice start must be numeric or undefined")
        }
        if !(stop is NumericValue || stop is UndefinedValue) {
            throw JinjaError.runtime("Slice stop must be numeric or undefined")
        }
        if !(step is NumericValue || step is UndefinedValue) {
            throw JinjaError.runtime("Slice step must be numeric or undefined")
        }
        if let object = object as? ArrayValue {
            return ArrayValue(
                value: slice(
                    object.value,
                    start: (start as? NumericValue)?.value as? Int,
                    stop: (stop as? NumericValue)?.value as? Int,
                    step: (step as? NumericValue)?.value as? Int
                )
            )
        } else if let object = object as? StringValue {
            return StringValue(
                value: slice(
                    Array(object.value),
                    start: (start as? NumericValue)?.value as? Int,
                    stop: (stop as? NumericValue)?.value as? Int,
                    step: (step as? NumericValue)?.value as? Int
                ).map { String($0) }.joined()
            )
        }
        throw JinjaError.runtime("Slice object must be an array or string")
    }

    func evaluateMemberExpression(expr: MemberExpression, environment: Environment) throws -> any RuntimeValue {
        let object = try self.evaluate(statement: expr.object, environment: environment)
        var property: any RuntimeValue
        if expr.computed {
            if let property = expr.property as? SliceExpression {
                return try self.evaluateSliceExpression(object: object, expr: property, environment: environment)
            } else {
                property = try self.evaluate(statement: expr.property, environment: environment)
            }
        } else {
            property = StringValue(value: (expr.property as! Identifier).value)
        }
        var value: (any RuntimeValue)?
        if let object = object as? ObjectValue {
            if let property = property as? StringValue {
                value = object.value[property.value] ?? object.builtins[property.value]
            } else {
                throw JinjaError.runtime("Cannot access property with non-string: got \(type(of:property))")
            }
        } else if let object = object as? ArrayValue {
            if let property = property as? NumericValue {
                if let index = property.value as? Int {
                    let actualIndex = index < 0 ? object.value.count + index : index
                    if actualIndex >= 0 && actualIndex < object.value.count {
                        value = object.value[actualIndex]
                    } else {
                        value = UndefinedValue()
                    }
                } else {
                    throw JinjaError.runtime("Array index must be an integer")
                }
            } else if let property = property as? StringValue {
                value = object.builtins[property.value]
            } else {
                throw JinjaError.runtime(
                    "Cannot access property with non-string/non-number: got \(type(of: property))"
                )
            }
        } else if let object = object as? StringValue {
            if let property = property as? NumericValue {
                if let index = property.value as? Int {
                    if index >= 0 && index < object.value.count {
                        let strIndex = object.value.index(object.value.startIndex, offsetBy: index)
                        value = StringValue(value: String(object.value[strIndex]))
                    } else if index < 0 && index >= -object.value.count {
                        let strIndex = object.value.index(object.value.startIndex, offsetBy: object.value.count + index)
                        value = StringValue(value: String(object.value[strIndex]))
                    } else {
                        value = UndefinedValue()
                    }
                } else {
                    throw JinjaError.runtime("String index must be an integer")
                }
            } else if let property = property as? StringValue {
                value = object.builtins[property.value]
            } else {
                throw JinjaError.runtime(
                    "Cannot access property with non-string/non-number: got \(type(of: property))"
                )
            }
        } else {
            if let property = property as? StringValue {
                value = object.builtins[property.value]
            } else {
                throw JinjaError.runtime("Cannot access property with non-string: got \(type(of:property))")
            }
        }
        if let value {
            return value
        } else {
            return UndefinedValue()
        }
    }

    func evaluateUnaryExpression(node: UnaryExpression, environment: Environment) throws -> any RuntimeValue {
        let argument = try self.evaluate(statement: node.argument, environment: environment)
        switch node.operation.value {
        case "not":
            return BooleanValue(value: !argument.bool())
        default:
            throw JinjaError.syntax("Unknown operator: \(node.operation.value)")
        }
    }

    func evaluateCallExpression(expr: CallExpression, environment: Environment) throws -> any RuntimeValue {
        var args: [any RuntimeValue] = []
        var kwargs: [String: any RuntimeValue] = [:]
        for argument in expr.args {
            if let argument = argument as? KeywordArgumentExpression {
                kwargs[argument.key.value] = try self.evaluate(statement: argument.value, environment: environment)
            } else {
                try args.append(self.evaluate(statement: argument, environment: environment))
            }
        }
        if !kwargs.isEmpty {
            args.append(ObjectValue(value: kwargs))
        }
        let fn = try self.evaluate(statement: expr.callee, environment: environment)
        if let fn = fn as? FunctionValue {
            return try fn.value(args, environment)
        } else {
            throw JinjaError.runtime("Cannot call something that is not a function: got \(type(of:fn))")
        }
    }

    func evaluateFilterExpression(node: FilterExpression, environment: Environment, whitespaceControl: Bool) throws
        -> any RuntimeValue
    {
        let operand = try self.evaluate(statement: node.operand, environment: environment)
        let filterName = node.filter.value
        guard let filter = environment.filters[filterName] else {
            throw JinjaError.runtime("No filter named '\(filterName)'")
        }
        // Evaluate positional arguments
        let evaluatedPositionalArgs = try node.args.map { arg in
            try self.evaluate(statement: arg, environment: environment)
        }
        // Create args array starting with operand
        var args: [any RuntimeValue] = [operand]
        args.append(contentsOf: evaluatedPositionalArgs)
        // If we have keyword arguments, add them as a final ObjectValue argument
        if !node.kwargs.isEmpty {
            var kwargs: [String: any RuntimeValue] = [:]
            for kwarg in node.kwargs {
                kwargs[kwarg.key.value] = try self.evaluate(statement: kwarg.value, environment: environment)
            }
            args.append(ObjectValue(value: kwargs))
        }
        return try filter(args, environment)
    }

    func evaluateTestExpression(node: TestExpression, environment: Environment) throws -> any RuntimeValue {
        let operand = try self.evaluate(statement: node.operand, environment: environment)
        guard let testFunction = environment.tests[node.test.value] else {
            throw JinjaError.runtime("Unknown test: \(node.test.value)")
        }
        let result = try testFunction([operand])
        return BooleanValue(value: node.negate ? !result : result)
    }

    func evaluateMacro(node: Macro, environment: Environment) throws -> NullValue {
        try environment.setVariable(
            name: node.name.value,
            value: FunctionValue(value: { args, scope in
                let macroScope = Environment(parent: scope)
                var args = args
                var kwargs: [String: any RuntimeValue] = [:]
                if let lastArg = args.last, let keywordArgsValue = lastArg as? KeywordArgumentsValue {
                    kwargs = keywordArgsValue.value
                    args.removeLast()
                }
                for i in 0 ..< node.args.count {
                    let nodeArg = node.args[i]
                    let passedArg = args.count > i ? args[i] : nil

                    if let identifier = nodeArg as? Identifier {
                        if passedArg == nil {
                            if let defaultValue = kwargs[identifier.value] {
                                try macroScope.setVariable(name: identifier.value, value: defaultValue)
                            } else {
                                throw JinjaError.runtime("Missing argument: \(identifier.value)")
                            }
                        } else {
                            try macroScope.setVariable(name: identifier.value, value: passedArg!)
                        }
                    } else if let kwarg = nodeArg as? KeywordArgumentExpression {
                        let value =
                            try kwargs[kwarg.key.value]
                            ?? (passedArg ?? (try self.evaluate(statement: kwarg.value, environment: macroScope)))

                        try macroScope.setVariable(name: kwarg.key.value, value: value)
                    } else {
                        throw JinjaError.runtime("Unknown argument type: \(type(of: nodeArg))")
                    }
                }
                return try self.evaluateBlock(statements: node.body, environment: macroScope)
            })
        )
        return NullValue()
    }

    func evaluateArguments(
        args: [Expression],
        environment: Environment
    ) throws -> ([any RuntimeValue], [String: any RuntimeValue]) {
        var positionalArguments: [any RuntimeValue] = []
        var keywordArguments: [String: any RuntimeValue] = [:]
        for argument in args {
            if let keywordArgument = argument as? KeywordArgumentExpression {
                keywordArguments[keywordArgument.key.value] = try self.evaluate(
                    statement: keywordArgument.value,
                    environment: environment
                )
            } else {
                if !keywordArguments.isEmpty {
                    throw JinjaError.runtime("Positional arguments must come before keyword arguments")
                }
                positionalArguments.append(try self.evaluate(statement: argument, environment: environment))
            }
        }

        return (positionalArguments, keywordArguments)
    }

    func evaluate(statement: Statement?, environment: Environment, whitespaceControl: Bool = false) throws
        -> any RuntimeValue
    {
        if let statement {
            switch statement {
            case let statement as Program:
                return try self.evalProgram(program: statement, environment: environment)
            case let statement as If:
                return try self.evaluateIf(node: statement, environment: environment)
            case let statement as StringLiteral:
                return StringValue(value: statement.value)
            case let statement as Set:
                return try self.evaluateSet(node: statement, environment: environment)
            case let statement as For:
                return try self.evaluateFor(node: statement, environment: environment)
            case let statement as Identifier:
                return try self.evaluateIdentifier(node: statement, environment: environment)
            case let statement as BinaryExpression:
                return try self.evaluateBinaryExpression(node: statement, environment: environment)
            case let statement as MemberExpression:
                return try self.evaluateMemberExpression(expr: statement, environment: environment)
            case let statement as UnaryExpression:
                return try self.evaluateUnaryExpression(node: statement, environment: environment)
            case let statement as NumericLiteral:
                if let intValue = statement.value as? Int {
                    return NumericValue(value: intValue)
                } else if let doubleValue = statement.value as? Double {
                    return NumericValue(value: doubleValue)
                } else {
                    throw JinjaError.runtime("Invalid numeric literal value")
                }
            case let statement as CallExpression:
                return try self.evaluateCallExpression(expr: statement, environment: environment)
            case let statement as BoolLiteral:
                return BooleanValue(value: statement.value)
            case let statement as FilterExpression:
                return try self.evaluateFilterExpression(
                    node: statement,
                    environment: environment,
                    whitespaceControl: whitespaceControl
                )
            case let statement as TestExpression:
                return try self.evaluateTestExpression(node: statement, environment: environment)
            case let statement as ArrayLiteral:
                return ArrayValue(
                    value: try statement.value.map { try self.evaluate(statement: $0, environment: environment) }
                )
            case let statement as TupleLiteral:
                return TupleValue(
                    value: try statement.value.map { try self.evaluate(statement: $0, environment: environment) }
                )
            case let statement as ObjectLiteral:
                var mapping: [String: any RuntimeValue] = [:]
                for (key, value) in statement.value {
                    mapping[key] = try self.evaluate(statement: value, environment: environment)
                }
                return ObjectValue(value: mapping)
            case let statement as Macro:
                return try self.evaluateMacro(node: statement, environment: environment)
            case is NullLiteral:
                return NullValue()
            default:
                throw JinjaError.runtime(
                    "Unknown node type: \(type(of:statement)), statement: \(String(describing: statement))"
                )
            }
        } else {
            return UndefinedValue()
        }
    }
}
