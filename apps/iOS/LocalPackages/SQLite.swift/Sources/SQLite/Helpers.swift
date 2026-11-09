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

#if SQLITE_SWIFT_STANDALONE
import sqlite3
#elseif SQLITE_SWIFT_SQLCIPHER
import SQLCipher
#elseif os(Linux)
import CSQLite
#else
import SQLite3
#endif

public typealias Star = (Expression<Binding>?, Expression<Binding>?) -> Expression<Void>

public func *(_: Expression<Binding>?, _: Expression<Binding>?) -> Expression<Void> {
    Expression(literal: "*")
}

// swiftlint:disable:next type_name
public protocol _OptionalType {

    associatedtype WrappedType

}

extension Optional: _OptionalType {

    public typealias WrappedType = Wrapped

}

// let SQLITE_STATIC = unsafeBitCast(0, sqlite3_destructor_type.self)
let SQLITE_TRANSIENT = unsafeBitCast(-1, to: sqlite3_destructor_type.self)

extension String {
    func quote(_ mark: Character = "\"") -> String {
        var quoted = ""
        quoted.append(mark)
        for character in self {
            quoted.append(character)
            if character == mark {
                quoted.append(character)
            }
        }
        quoted.append(mark)
        return quoted
    }

    func join(_ expressions: [Expressible]) -> Expressible {
        var (template, bindings) = ([String](), [Binding?]())
        for expressible in expressions {
            let expression = expressible.expression
            template.append(expression.template)
            bindings.append(contentsOf: expression.bindings)
        }
        return Expression<Void>(template.joined(separator: self), bindings)
    }

    func infix<T>(_ lhs: Expressible, _ rhs: Expressible, wrap: Bool = true) -> Expression<T> {
        infix([lhs, rhs], wrap: wrap)
    }

    func infix<T>(_ terms: [Expressible], wrap: Bool = true) -> Expression<T> {
        let expression = Expression<T>(" \(self) ".join(terms).expression)
        guard wrap else {
            return expression
        }
        return "".wrap(expression)
    }

    func prefix(_ expressions: Expressible) -> Expressible {
        "\(self) ".wrap(expressions) as Expression<Void>
    }

    func prefix(_ expressions: [Expressible]) -> Expressible {
        "\(self) ".wrap(expressions) as Expression<Void>
    }

    func wrap<T>(_ expression: Expressible) -> Expression<T> {
        Expression("\(self)(\(expression.expression.template))", expression.expression.bindings)
    }

    func wrap<T>(_ expressions: [Expressible]) -> Expression<T> {
        wrap(", ".join(expressions))
    }

}

func transcode(_ literal: Binding?) -> String {
    guard let literal else { return "NULL" }

    switch literal {
    case let blob as Blob:
        return blob.description
    case let string as String:
        return string.quote("'")
    case let binding:
        return "\(binding)"
    }
}

// swiftlint:disable force_cast force_try
func value<A: Value>(_ binding: Binding) -> A {
    try! A.fromDatatypeValue(binding as! A.Datatype) as! A
}

func value<A: Value>(_ binding: Binding?) -> A {
    value(binding!)
}
