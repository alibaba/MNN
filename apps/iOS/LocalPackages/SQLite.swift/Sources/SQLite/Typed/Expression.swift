//
// SQLite.swift
// https://github.com/stephencelis/SQLite.swift
// Copyright © 2014-2015 Stephen Celis.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//

public protocol ExpressionType: Expressible, CustomStringConvertible { // extensions cannot have inheritance clauses

    associatedtype UnderlyingType = Void

    var template: String { get }
    var bindings: [Binding?] { get }

    init(_ template: String, _ bindings: [Binding?])

}

extension ExpressionType {

    public init(literal: String) {
        self.init(literal, [])
    }

    public init(_ identifier: String) {
        self.init(literal: identifier.quote())
    }

    public init<U: ExpressionType>(_ expression: U) {
        self.init(expression.template, expression.bindings)
    }

    public var description: String {
        asSQL()
    }
}

/// An `Expression` represents a raw SQL fragment and any associated bindings.
public struct Expression<Datatype>: ExpressionType {

    public typealias UnderlyingType = Datatype

    public var template: String
    public var bindings: [Binding?]

    public init(_ template: String, _ bindings: [Binding?]) {
        self.template = template
        self.bindings = bindings
    }

}

public protocol Expressible {

    var expression: Expression<Void> { get }

}

extension Expressible {

    // naïve compiler for statements that can’t be bound, e.g., CREATE TABLE
    func asSQL() -> String {
        let expressed = expression
        return expressed.template.reduce(("", 0)) { memo, character in
            let (template, index) = memo

            if character == "?" {
                precondition(index < expressed.bindings.count, "not enough bindings for expression")
                return (template + transcode(expressed.bindings[index]), index + 1)
            } else {
                return (template + String(character), index)
            }
        }.0
    }
}

extension ExpressionType {

    public var expression: Expression<Void> {
        Expression(template, bindings)
    }

    public var asc: Expressible {
        " ".join([self, Expression<Void>(literal: "ASC")])
    }

    public var desc: Expressible {
        " ".join([self, Expression<Void>(literal: "DESC")])
    }

}

extension ExpressionType where UnderlyingType: Value {

    public init(value: UnderlyingType) {
        self.init("?", [value.datatypeValue])
    }

}

extension ExpressionType where UnderlyingType: _OptionalType, UnderlyingType.WrappedType: Value {

    public static var null: Self {
        self.init(value: nil)
    }

    public init(value: UnderlyingType.WrappedType?) {
        self.init("?", [value?.datatypeValue])
    }

}

extension Value {

    public var expression: Expression<Void> {
        Expression(value: self).expression
    }

}

public let rowid = Expression<Int64>("ROWID")

public func cast<T: Value, U: Value>(_ expression: Expression<T>) -> Expression<U> {
    Expression("CAST (\(expression.template) AS \(U.declaredDatatype))", expression.bindings)
}

public func cast<T: Value, U: Value>(_ expression: Expression<T?>) -> Expression<U?> {
    Expression("CAST (\(expression.template) AS \(U.declaredDatatype))", expression.bindings)
}
