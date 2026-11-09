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
import Foundation

private enum Function: String {
    case abs
    case round
    case random
    case randomblob
    case zeroblob
    case length
    case lower
    case upper
    case ltrim
    case rtrim
    case trim
    case replace
    case substr
    case like = "LIKE"
    case `in` = "IN"
    case glob = "GLOB"
    case match = "MATCH"
    case regexp = "REGEXP"
    case collate = "COLLATE"
    case ifnull

    func infix<T>(_ lhs: Expressible, _ rhs: Expressible, wrap: Bool = true) -> Expression<T> {
        self.rawValue.infix(lhs, rhs, wrap: wrap)
    }

    func wrap<T>(_ expression: Expressible) -> Expression<T> {
        self.rawValue.wrap(expression)
    }

    func wrap<T>(_ expressions: [Expressible]) -> Expression<T> {
        self.rawValue.wrap(", ".join(expressions))
    }
}

extension ExpressionType where UnderlyingType: Number {

    /// Builds a copy of the expression wrapped with the `abs` function.
    ///
    ///     let x = Expression<Int>("x")
    ///     x.absoluteValue
    ///     // abs("x")
    ///
    /// - Returns: A copy of the expression wrapped with the `abs` function.
    public var absoluteValue: Expression<UnderlyingType> {
        Function.abs.wrap(self)
    }

}

extension ExpressionType where UnderlyingType: _OptionalType, UnderlyingType.WrappedType: Number {

    /// Builds a copy of the expression wrapped with the `abs` function.
    ///
    ///     let x = Expression<Int?>("x")
    ///     x.absoluteValue
    ///     // abs("x")
    ///
    /// - Returns: A copy of the expression wrapped with the `abs` function.
    public var absoluteValue: Expression<UnderlyingType> {
        Function.abs.wrap(self)
    }

}

extension ExpressionType where UnderlyingType == Double {

    /// Builds a copy of the expression wrapped with the `round` function.
    ///
    ///     let salary = Expression<Double>("salary")
    ///     salary.round()
    ///     // round("salary")
    ///     salary.round(2)
    ///     // round("salary", 2)
    ///
    /// - Returns: A copy of the expression wrapped with the `round` function.
    public func round(_ precision: Int? = nil) -> Expression<UnderlyingType> {
        guard let precision else {
            return Function.round.wrap([self])
        }
        return Function.round.wrap([self, Int(precision)])
    }

}

extension ExpressionType where UnderlyingType == Double? {

    /// Builds a copy of the expression wrapped with the `round` function.
    ///
    ///     let salary = Expression<Double>("salary")
    ///     salary.round()
    ///     // round("salary")
    ///     salary.round(2)
    ///     // round("salary", 2)
    ///
    /// - Returns: A copy of the expression wrapped with the `round` function.
    public func round(_ precision: Int? = nil) -> Expression<UnderlyingType> {
        guard let precision else {
            return Function.round.wrap(self)
        }
        return Function.round.wrap([self, Int(precision)])
    }

}

extension ExpressionType where UnderlyingType: Value, UnderlyingType.Datatype == Int64 {

    /// Builds an expression representing the `random` function.
    ///
    ///     Expression<Int>.random()
    ///     // random()
    ///
    /// - Returns: An expression calling the `random` function.
    public static func random() -> Expression<UnderlyingType> {
        Function.random.wrap([])
    }

}

extension ExpressionType where UnderlyingType == Data {

    /// Builds an expression representing the `randomblob` function.
    ///
    ///     Expression<Int>.random(16)
    ///     // randomblob(16)
    ///
    /// - Parameter length: Length in bytes.
    ///
    /// - Returns: An expression calling the `randomblob` function.
    public static func random(_ length: Int) -> Expression<UnderlyingType> {
        Function.randomblob.wrap([])
    }

    /// Builds an expression representing the `zeroblob` function.
    ///
    ///     Expression<Int>.allZeros(16)
    ///     // zeroblob(16)
    ///
    /// - Parameter length: Length in bytes.
    ///
    /// - Returns: An expression calling the `zeroblob` function.
    public static func allZeros(_ length: Int) -> Expression<UnderlyingType> {
        Function.zeroblob.wrap([])
    }

    /// Builds a copy of the expression wrapped with the `length` function.
    ///
    ///     let data = Expression<NSData>("data")
    ///     data.length
    ///     // length("data")
    ///
    /// - Returns: A copy of the expression wrapped with the `length` function.
    public var length: Expression<Int> {
        Function.length.wrap(self)
    }

}

extension ExpressionType where UnderlyingType == Data? {

    /// Builds a copy of the expression wrapped with the `length` function.
    ///
    ///     let data = Expression<NSData?>("data")
    ///     data.length
    ///     // length("data")
    ///
    /// - Returns: A copy of the expression wrapped with the `length` function.
    public var length: Expression<Int?> {
        Function.length.wrap(self)
    }

}

extension ExpressionType where UnderlyingType == String {

    /// Builds a copy of the expression wrapped with the `length` function.
    ///
    ///     let name = Expression<String>("name")
    ///     name.length
    ///     // length("name")
    ///
    /// - Returns: A copy of the expression wrapped with the `length` function.
    public var length: Expression<Int> {
        Function.length.wrap(self)
    }

    /// Builds a copy of the expression wrapped with the `lower` function.
    ///
    ///     let name = Expression<String>("name")
    ///     name.lowercaseString
    ///     // lower("name")
    ///
    /// - Returns: A copy of the expression wrapped with the `lower` function.
    public var lowercaseString: Expression<UnderlyingType> {
        Function.lower.wrap(self)
    }

    /// Builds a copy of the expression wrapped with the `upper` function.
    ///
    ///     let name = Expression<String>("name")
    ///     name.uppercaseString
    ///     // upper("name")
    ///
    /// - Returns: A copy of the expression wrapped with the `upper` function.
    public var uppercaseString: Expression<UnderlyingType> {
        Function.upper.wrap(self)
    }

    /// Builds a copy of the expression appended with a `LIKE` query against the
    /// given pattern.
    ///
    ///     let email = Expression<String>("email")
    ///     email.like("%@example.com")
    ///     // "email" LIKE '%@example.com'
    ///     email.like("99\\%@%", escape: "\\")
    ///     // "email" LIKE '99\%@%' ESCAPE '\'
    ///
    /// - Parameters:
    ///
    ///   - pattern: A pattern to match.
    ///
    ///   - escape: An (optional) character designated for escaping
    ///     pattern-matching characters (*i.e.*, the `%` and `_` characters).
    ///
    /// - Returns: A copy of the expression appended with a `LIKE` query against
    ///   the given pattern.
    public func like(_ pattern: String, escape character: Character? = nil) -> Expression<Bool> {
        guard let character else {
            return "LIKE".infix(self, pattern)
        }
        return Expression("(\(template) LIKE ? ESCAPE ?)", bindings + [pattern, String(character)])
    }

    /// Builds a copy of the expression appended with a `LIKE` query against the
    /// given pattern.
    ///
    ///     let email = Expression<String>("email")
    ///     let pattern = Expression<String>("pattern")
    ///     email.like(pattern)
    ///     // "email" LIKE "pattern"
    ///
    /// - Parameters:
    ///
    ///   - pattern: A pattern to match.
    ///
    ///   - escape: An (optional) character designated for escaping
    ///     pattern-matching characters (*i.e.*, the `%` and `_` characters).
    ///
    /// - Returns: A copy of the expression appended with a `LIKE` query against
    ///   the given pattern.
    public func like(_ pattern: Expression<String>, escape character: Character? = nil) -> Expression<Bool> {
        guard let character else {
            return Function.like.infix(self, pattern)
        }
        let like: Expression<Bool> =  Function.like.infix(self, pattern, wrap: false)
        return Expression("(\(like.template) ESCAPE ?)", like.bindings + [String(character)])
    }

    /// Builds a copy of the expression appended with a `GLOB` query against the
    /// given pattern.
    ///
    ///     let path = Expression<String>("path")
    ///     path.glob("*.png")
    ///     // "path" GLOB '*.png'
    ///
    /// - Parameter pattern: A pattern to match.
    ///
    /// - Returns: A copy of the expression appended with a `GLOB` query against
    ///   the given pattern.
    public func glob(_ pattern: String) -> Expression<Bool> {
        Function.glob.infix(self, pattern)
    }

    /// Builds a copy of the expression appended with a `MATCH` query against
    /// the given pattern.
    ///
    ///     let title = Expression<String>("title")
    ///     title.match("swift AND programming")
    ///     // "title" MATCH 'swift AND programming'
    ///
    /// - Parameter pattern: A pattern to match.
    ///
    /// - Returns: A copy of the expression appended with a `MATCH` query
    ///   against the given pattern.
    public func match(_ pattern: String) -> Expression<Bool> {
        Function.match.infix(self, pattern)
    }

    /// Builds a copy of the expression appended with a `REGEXP` query against
    /// the given pattern.
    ///
    /// - Parameter pattern: A pattern to match.
    ///
    /// - Returns: A copy of the expression appended with a `REGEXP` query
    ///   against the given pattern.
    public func regexp(_ pattern: String) -> Expression<Bool> {
        Function.regexp.infix(self, pattern)
    }

    /// Builds a copy of the expression appended with a `COLLATE` clause with
    /// the given sequence.
    ///
    ///     let name = Expression<String>("name")
    ///     name.collate(.Nocase)
    ///     // "name" COLLATE NOCASE
    ///
    /// - Parameter collation: A collating sequence.
    ///
    /// - Returns: A copy of the expression appended with a `COLLATE` clause
    ///   with the given sequence.
    public func collate(_ collation: Collation) -> Expression<UnderlyingType> {
        Function.collate.infix(self, collation)
    }

    /// Builds a copy of the expression wrapped with the `ltrim` function.
    ///
    ///     let name = Expression<String>("name")
    ///     name.ltrim()
    ///     // ltrim("name")
    ///     name.ltrim([" ", "\t"])
    ///     // ltrim("name", ' \t')
    ///
    /// - Parameter characters: A set of characters to trim.
    ///
    /// - Returns: A copy of the expression wrapped with the `ltrim` function.
    public func ltrim(_ characters: Set<Character>? = nil) -> Expression<UnderlyingType> {
        guard let characters else {
            return Function.ltrim.wrap(self)
        }
        return Function.ltrim.wrap([self, String(characters)])
    }

    /// Builds a copy of the expression wrapped with the `rtrim` function.
    ///
    ///     let name = Expression<String>("name")
    ///     name.rtrim()
    ///     // rtrim("name")
    ///     name.rtrim([" ", "\t"])
    ///     // rtrim("name", ' \t')
    ///
    /// - Parameter characters: A set of characters to trim.
    ///
    /// - Returns: A copy of the expression wrapped with the `rtrim` function.
    public func rtrim(_ characters: Set<Character>? = nil) -> Expression<UnderlyingType> {
        guard let characters else {
            return Function.rtrim.wrap(self)
        }
        return Function.rtrim.wrap([self, String(characters)])
    }

    /// Builds a copy of the expression wrapped with the `trim` function.
    ///
    ///     let name = Expression<String>("name")
    ///     name.trim()
    ///     // trim("name")
    ///     name.trim([" ", "\t"])
    ///     // trim("name", ' \t')
    ///
    /// - Parameter characters: A set of characters to trim.
    ///
    /// - Returns: A copy of the expression wrapped with the `trim` function.
    public func trim(_ characters: Set<Character>? = nil) -> Expression<UnderlyingType> {
        guard let characters else {
            return Function.trim.wrap([self])
        }
        return Function.trim.wrap([self, String(characters)])
    }

    /// Builds a copy of the expression wrapped with the `replace` function.
    ///
    ///     let email = Expression<String>("email")
    ///     email.replace("@mac.com", with: "@icloud.com")
    ///     // replace("email", '@mac.com', '@icloud.com')
    ///
    /// - Parameters:
    ///
    ///   - pattern: A pattern to match.
    ///
    ///   - replacement: The replacement string.
    ///
    /// - Returns: A copy of the expression wrapped with the `replace` function.
    public func replace(_ pattern: String, with replacement: String) -> Expression<UnderlyingType> {
        Function.replace.wrap([self, pattern, replacement])
    }

    public func substring(_ location: Int, length: Int? = nil) -> Expression<UnderlyingType> {
        guard let length else {
            return Function.substr.wrap([self, location])
        }
        return Function.substr.wrap([self, location, length])
    }

    public subscript(range: Range<Int>) -> Expression<UnderlyingType> {
        substring(range.lowerBound, length: range.upperBound - range.lowerBound)
    }

}

extension ExpressionType where UnderlyingType == String? {

    /// Builds a copy of the expression wrapped with the `length` function.
    ///
    ///     let name = Expression<String?>("name")
    ///     name.length
    ///     // length("name")
    ///
    /// - Returns: A copy of the expression wrapped with the `length` function.
    public var length: Expression<Int?> {
        Function.length.wrap(self)
    }

    /// Builds a copy of the expression wrapped with the `lower` function.
    ///
    ///     let name = Expression<String?>("name")
    ///     name.lowercaseString
    ///     // lower("name")
    ///
    /// - Returns: A copy of the expression wrapped with the `lower` function.
    public var lowercaseString: Expression<UnderlyingType> {
        Function.lower.wrap(self)
    }

    /// Builds a copy of the expression wrapped with the `upper` function.
    ///
    ///     let name = Expression<String?>("name")
    ///     name.uppercaseString
    ///     // lower("name")
    ///
    /// - Returns: A copy of the expression wrapped with the `upper` function.
    public var uppercaseString: Expression<UnderlyingType> {
        Function.upper.wrap(self)
    }

    /// Builds a copy of the expression appended with a `LIKE` query against the
    /// given pattern.
    ///
    ///     let email = Expression<String?>("email")
    ///     email.like("%@example.com")
    ///     // "email" LIKE '%@example.com'
    ///     email.like("99\\%@%", escape: "\\")
    ///     // "email" LIKE '99\%@%' ESCAPE '\'
    ///
    /// - Parameters:
    ///
    ///   - pattern: A pattern to match.
    ///
    ///   - escape: An (optional) character designated for escaping
    ///     pattern-matching characters (*i.e.*, the `%` and `_` characters).
    ///
    /// - Returns: A copy of the expression appended with a `LIKE` query against
    ///   the given pattern.
    public func like(_ pattern: String, escape character: Character? = nil) -> Expression<Bool?> {
        guard let character else {
            return Function.like.infix(self, pattern)
        }
        return Expression("(\(template) LIKE ? ESCAPE ?)", bindings + [pattern, String(character)])
    }

    /// Builds a copy of the expression appended with a `LIKE` query against the
    /// given pattern.
    ///
    ///     let email = Expression<String>("email")
    ///     let pattern = Expression<String>("pattern")
    ///     email.like(pattern)
    ///     // "email" LIKE "pattern"
    ///
    /// - Parameters:
    ///
    ///   - pattern: A pattern to match.
    ///
    ///   - escape: An (optional) character designated for escaping
    ///     pattern-matching characters (*i.e.*, the `%` and `_` characters).
    ///
    /// - Returns: A copy of the expression appended with a `LIKE` query against
    ///   the given pattern.
    public func like(_ pattern: Expression<String>, escape character: Character? = nil) -> Expression<Bool?> {
        guard let character else {
            return Function.like.infix(self, pattern)
        }
        let like: Expression<Bool> = Function.like.infix(self, pattern, wrap: false)
        return Expression("(\(like.template) ESCAPE ?)", like.bindings + [String(character)])
    }

    /// Builds a copy of the expression appended with a `GLOB` query against the
    /// given pattern.
    ///
    ///     let path = Expression<String?>("path")
    ///     path.glob("*.png")
    ///     // "path" GLOB '*.png'
    ///
    /// - Parameter pattern: A pattern to match.
    ///
    /// - Returns: A copy of the expression appended with a `GLOB` query against
    ///   the given pattern.
    public func glob(_ pattern: String) -> Expression<Bool?> {
        Function.glob.infix(self, pattern)
    }

    /// Builds a copy of the expression appended with a `MATCH` query against
    /// the given pattern.
    ///
    ///     let title = Expression<String?>("title")
    ///     title.match("swift AND programming")
    ///     // "title" MATCH 'swift AND programming'
    ///
    /// - Parameter pattern: A pattern to match.
    ///
    /// - Returns: A copy of the expression appended with a `MATCH` query
    ///   against the given pattern.
    public func match(_ pattern: String) -> Expression<Bool> {
        Function.match.infix(self, pattern)
    }

    /// Builds a copy of the expression appended with a `REGEXP` query against
    /// the given pattern.
    ///
    /// - Parameter pattern: A pattern to match.
    ///
    /// - Returns: A copy of the expression appended with a `REGEXP` query
    ///   against the given pattern.
    public func regexp(_ pattern: String) -> Expression<Bool?> {
        Function.regexp.infix(self, pattern)
    }

    /// Builds a copy of the expression appended with a `COLLATE` clause with
    /// the given sequence.
    ///
    ///     let name = Expression<String?>("name")
    ///     name.collate(.Nocase)
    ///     // "name" COLLATE NOCASE
    ///
    /// - Parameter collation: A collating sequence.
    ///
    /// - Returns: A copy of the expression appended with a `COLLATE` clause
    ///   with the given sequence.
    public func collate(_ collation: Collation) -> Expression<UnderlyingType> {
        Function.collate.infix(self, collation)
    }

    /// Builds a copy of the expression wrapped with the `ltrim` function.
    ///
    ///     let name = Expression<String?>("name")
    ///     name.ltrim()
    ///     // ltrim("name")
    ///     name.ltrim([" ", "\t"])
    ///     // ltrim("name", ' \t')
    ///
    /// - Parameter characters: A set of characters to trim.
    ///
    /// - Returns: A copy of the expression wrapped with the `ltrim` function.
    public func ltrim(_ characters: Set<Character>? = nil) -> Expression<UnderlyingType> {
        guard let characters else {
            return Function.ltrim.wrap(self)
        }
        return Function.ltrim.wrap([self, String(characters)])
    }

    /// Builds a copy of the expression wrapped with the `rtrim` function.
    ///
    ///     let name = Expression<String?>("name")
    ///     name.rtrim()
    ///     // rtrim("name")
    ///     name.rtrim([" ", "\t"])
    ///     // rtrim("name", ' \t')
    ///
    /// - Parameter characters: A set of characters to trim.
    ///
    /// - Returns: A copy of the expression wrapped with the `rtrim` function.
    public func rtrim(_ characters: Set<Character>? = nil) -> Expression<UnderlyingType> {
        guard let characters else {
            return Function.rtrim.wrap(self)
        }
        return Function.rtrim.wrap([self, String(characters)])
    }

    /// Builds a copy of the expression wrapped with the `trim` function.
    ///
    ///     let name = Expression<String?>("name")
    ///     name.trim()
    ///     // trim("name")
    ///     name.trim([" ", "\t"])
    ///     // trim("name", ' \t')
    ///
    /// - Parameter characters: A set of characters to trim.
    ///
    /// - Returns: A copy of the expression wrapped with the `trim` function.
    public func trim(_ characters: Set<Character>? = nil) -> Expression<UnderlyingType> {
        guard let characters else {
            return Function.trim.wrap(self)
        }
        return Function.trim.wrap([self, String(characters)])
    }

    /// Builds a copy of the expression wrapped with the `replace` function.
    ///
    ///     let email = Expression<String?>("email")
    ///     email.replace("@mac.com", with: "@icloud.com")
    ///     // replace("email", '@mac.com', '@icloud.com')
    ///
    /// - Parameters:
    ///
    ///   - pattern: A pattern to match.
    ///
    ///   - replacement: The replacement string.
    ///
    /// - Returns: A copy of the expression wrapped with the `replace` function.
    public func replace(_ pattern: String, with replacement: String) -> Expression<UnderlyingType> {
        Function.replace.wrap([self, pattern, replacement])
    }

    /// Builds a copy of the expression wrapped with the `substr` function.
    ///
    ///     let title = Expression<String?>("title")
    ///     title.substr(-100)
    ///     // substr("title", -100)
    ///     title.substr(0, length: 100)
    ///     // substr("title", 0, 100)
    ///
    /// - Parameters:
    ///
    ///   - location: The substring’s start index.
    ///
    ///   - length: An optional substring length.
    ///
    /// - Returns: A copy of the expression wrapped with the `substr` function.
    public func substring(_ location: Int, length: Int? = nil) -> Expression<UnderlyingType> {
        guard let length else {
            return Function.substr.wrap([self, location])
        }
        return Function.substr.wrap([self, location, length])
    }

    /// Builds a copy of the expression wrapped with the `substr` function.
    ///
    ///     let title = Expression<String?>("title")
    ///     title[0..<100]
    ///     // substr("title", 0, 100)
    ///
    /// - Parameter range: The character index range of the substring.
    ///
    /// - Returns: A copy of the expression wrapped with the `substr` function.
    public subscript(range: Range<Int>) -> Expression<UnderlyingType> {
        substring(range.lowerBound, length: range.upperBound - range.lowerBound)
    }

}

extension Collection where Iterator.Element: Value {

    /// Builds a copy of the expression prepended with an `IN` check against the
    /// collection.
    ///
    ///     let name = Expression<String>("name")
    ///     ["alice", "betty"].contains(name)
    ///     // "name" IN ('alice', 'betty')
    ///
    /// - Parameter pattern: A pattern to match.
    ///
    /// - Returns: A copy of the expression prepended with an `IN` check against
    ///   the collection.
    public func contains(_ expression: Expression<Iterator.Element>) -> Expression<Bool> {
        let templates = [String](repeating: "?", count: count).joined(separator: ", ")
        return Function.in.infix(expression, Expression<Void>("(\(templates))", map { $0.datatypeValue }))
    }

    /// Builds a copy of the expression prepended with an `IN` check against the
    /// collection.
    ///
    ///     let name = Expression<String?>("name")
    ///     ["alice", "betty"].contains(name)
    ///     // "name" IN ('alice', 'betty')
    ///
    /// - Parameter pattern: A pattern to match.
    ///
    /// - Returns: A copy of the expression prepended with an `IN` check against
    ///   the collection.
    public func contains(_ expression: Expression<Iterator.Element?>) -> Expression<Bool?> {
        let templates = [String](repeating: "?", count: count).joined(separator: ", ")
        return Function.in.infix(expression, Expression<Void>("(\(templates))", map { $0.datatypeValue }))
    }

}

extension String {

    /// Builds a copy of the expression appended with a `LIKE` query against the
    /// given pattern.
    ///
    ///     let email = "some@thing.com"
    ///     let pattern = Expression<String>("pattern")
    ///     email.like(pattern)
    ///     // 'some@thing.com' LIKE "pattern"
    ///
    /// - Parameters:
    ///
    ///   - pattern: A pattern to match.
    ///
    ///   - escape: An (optional) character designated for escaping
    ///     pattern-matching characters (*i.e.*, the `%` and `_` characters).
    ///
    /// - Returns: A copy of the expression appended with a `LIKE` query against
    ///   the given pattern.
    public func like(_ pattern: Expression<String>, escape character: Character? = nil) -> Expression<Bool> {
        guard let character else {
            return Function.like.infix(self, pattern)
        }
        let like: Expression<Bool> = Function.like.infix(self, pattern, wrap: false)
        return Expression("(\(like.template) ESCAPE ?)", like.bindings + [String(character)])
    }

}

/// Builds a copy of the given expressions wrapped with the `ifnull` function.
///
///     let name = Expression<String?>("name")
///     name ?? "An Anonymous Coward"
///     // ifnull("name", 'An Anonymous Coward')
///
/// - Parameters:
///
///   - optional: An optional expression.
///
///   - defaultValue: A fallback value for when the optional expression is
///     `nil`.
///
/// - Returns: A copy of the given expressions wrapped with the `ifnull`
///   function.
public func ??<V: Value>(optional: Expression<V?>, defaultValue: V) -> Expression<V> {
    Function.ifnull.wrap([optional, defaultValue])
}

/// Builds a copy of the given expressions wrapped with the `ifnull` function.
///
///     let nick = Expression<String?>("nick")
///     let name = Expression<String>("name")
///     nick ?? name
///     // ifnull("nick", "name")
///
/// - Parameters:
///
///   - optional: An optional expression.
///
///   - defaultValue: A fallback expression for when the optional expression is
///     `nil`.
///
/// - Returns: A copy of the given expressions wrapped with the `ifnull`
///   function.
public func ??<V: Value>(optional: Expression<V?>, defaultValue: Expression<V>) -> Expression<V> {
    Function.ifnull.wrap([optional, defaultValue])
}

/// Builds a copy of the given expressions wrapped with the `ifnull` function.
///
///     let nick = Expression<String?>("nick")
///     let name = Expression<String?>("name")
///     nick ?? name
///     // ifnull("nick", "name")
///
/// - Parameters:
///
///   - optional: An optional expression.
///
///   - defaultValue: A fallback expression for when the optional expression is
///     `nil`.
///
/// - Returns: A copy of the given expressions wrapped with the `ifnull`
///   function.
public func ??<V: Value>(optional: Expression<V?>, defaultValue: Expression<V?>) -> Expression<V> {
    Function.ifnull.wrap([optional, defaultValue])
}
