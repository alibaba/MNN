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

extension Module {

    public static func FTS4(_ column: Expressible, _ more: Expressible...) -> Module {
        FTS4([column] + more)
    }

    public static func FTS4(_ columns: [Expressible] = [], tokenize tokenizer: Tokenizer? = nil) -> Module {
        FTS4(FTS4Config().columns(columns).tokenizer(tokenizer))
    }

    public static func FTS4(_ config: FTS4Config) -> Module {
        Module(name: "fts4", arguments: config.arguments())
    }
}

extension VirtualTable {

    /// Builds an expression appended with a `MATCH` query against the given
    /// pattern.
    ///
    ///     let emails = VirtualTable("emails")
    ///
    ///     emails.filter(emails.match("Hello"))
    ///     // SELECT * FROM "emails" WHERE "emails" MATCH 'Hello'
    ///
    /// - Parameter pattern: A pattern to match.
    ///
    /// - Returns: An expression appended with a `MATCH` query against the given
    ///   pattern.
    public func match(_ pattern: String) -> Expression<Bool> {
        "MATCH".infix(tableName(), pattern)
    }

    public func match(_ pattern: Expression<String>) -> Expression<Bool> {
        "MATCH".infix(tableName(), pattern)
    }

    public func match(_ pattern: Expression<String?>) -> Expression<Bool?> {
        "MATCH".infix(tableName(), pattern)
    }

    /// Builds a copy of the query with a `WHERE … MATCH` clause.
    ///
    ///     let emails = VirtualTable("emails")
    ///
    ///     emails.match("Hello")
    ///     // SELECT * FROM "emails" WHERE "emails" MATCH 'Hello'
    ///
    /// - Parameter pattern: A pattern to match.
    ///
    /// - Returns: A query with the given `WHERE … MATCH` clause applied.
    public func match(_ pattern: String) -> QueryType {
        filter(match(pattern))
    }

    public func match(_ pattern: Expression<String>) -> QueryType {
        filter(match(pattern))
    }

    public func match(_ pattern: Expression<String?>) -> QueryType {
        filter(match(pattern))
    }

}

// swiftlint:disable identifier_name
public struct Tokenizer {

    public static let Simple = Tokenizer("simple")
    public static let Porter = Tokenizer("porter")

    public static func Unicode61(removeDiacritics: Bool? = nil,
                                 tokenchars: Set<Character> = [],
                                 separators: Set<Character> = []) -> Tokenizer {
        var arguments = [String]()

        if let removeDiacritics {
            arguments.append("remove_diacritics=\(removeDiacritics ? 1 : 0)".quote())
        }

        if !tokenchars.isEmpty {
            let joined = tokenchars.map { String($0) }.joined(separator: "")
            arguments.append("tokenchars=\(joined)".quote())
        }

        if !separators.isEmpty {
            let joined = separators.map { String($0) }.joined(separator: "")
            arguments.append("separators=\(joined)".quote())
        }

        return Tokenizer("unicode61", arguments)
    }

    // https://sqlite.org/fts5.html#the_experimental_trigram_tokenizer
    public static func Trigram(caseSensitive: Bool = false) -> Tokenizer {
        Tokenizer("trigram", ["case_sensitive", caseSensitive ? "1" : "0"])
    }

    public static func Custom(_ name: String) -> Tokenizer {
        Tokenizer(Tokenizer.moduleName.quote(), [name.quote()])
    }

    public let name: String

    public let arguments: [String]

    fileprivate init(_ name: String, _ arguments: [String] = []) {
        self.name = name
        self.arguments = arguments
    }

    fileprivate static let moduleName = "SQLite.swift"

}

extension Tokenizer: CustomStringConvertible {

    public var description: String {
        ([name] + arguments).joined(separator: " ")
    }

}

/// Configuration options shared between the [FTS4](https://www.sqlite.org/fts3.html) and
/// [FTS5](https://www.sqlite.org/fts5.html) extensions.
open class FTSConfig {
    public enum ColumnOption {
        /// [The notindexed= option](https://www.sqlite.org/fts3.html#section_6_5)
        case unindexed
    }

    typealias ColumnDefinition = (Expressible, options: [ColumnOption])
    var columnDefinitions = [ColumnDefinition]()
    var tokenizer: Tokenizer?
    var prefixes = [Int]()
    var externalContentSchema: SchemaType?
    var isContentless: Bool = false

    /// Adds a column definition
    @discardableResult open func column(_ column: Expressible, _ options: [ColumnOption] = []) -> Self {
        columnDefinitions.append((column, options))
        return self
    }

    @discardableResult open func columns(_ columns: [Expressible]) -> Self {
        for column in columns {
            self.column(column)
        }
        return self
    }

    /// [Tokenizers](https://www.sqlite.org/fts3.html#tokenizer)
    @discardableResult open func tokenizer(_ tokenizer: Tokenizer?) -> Self {
        self.tokenizer = tokenizer
        return self
    }

    /// [The prefix= option](https://www.sqlite.org/fts3.html#section_6_6)
    @discardableResult open func prefix(_ prefix: [Int]) -> Self {
        prefixes += prefix
        return self
    }

    /// [The content= option](https://www.sqlite.org/fts3.html#section_6_2)
    @discardableResult open func externalContent(_ schema: SchemaType) -> Self {
        externalContentSchema = schema
        return self
    }

    /// [Contentless FTS4 Tables](https://www.sqlite.org/fts3.html#section_6_2_1)
    @discardableResult open func contentless() -> Self {
        isContentless = true
        return self
    }

    func formatColumnDefinitions() -> [Expressible] {
        columnDefinitions.map { $0.0 }
    }

    func arguments() -> [Expressible] {
        options().arguments
    }

    func options() -> Options {
        var options = Options()
        options.append(formatColumnDefinitions())
        if let tokenizer {
            options.append("tokenize", value: Expression<Void>(literal: tokenizer.description))
        }
        options.appendCommaSeparated("prefix", values: prefixes.sorted().map { String($0) })
        if isContentless {
            options.append("content", value: "")
        } else if let externalContentSchema {
            options.append("content", value: externalContentSchema.tableName())
        }
        return options
    }

    struct Options {
        var arguments = [Expressible]()

        @discardableResult mutating func append(_ columns: [Expressible]) -> Options {
            arguments.append(contentsOf: columns)
            return self
        }

        @discardableResult mutating func appendCommaSeparated(_ key: String, values: [String]) -> Options {
            if values.isEmpty {
                return self
            } else {
                return append(key, value: values.joined(separator: ","))
            }
        }

        @discardableResult mutating func append(_ key: String, value: String) -> Options {
            append(key, value: Expression<String>(value))
        }

        @discardableResult mutating func append(_ key: String, value: Expressible) -> Options {
            arguments.append("=".join([Expression<Void>(literal: key), value]))
            return self
        }
    }
}

/// Configuration for the [FTS4](https://www.sqlite.org/fts3.html) extension.
open class FTS4Config: FTSConfig {
    /// [The matchinfo= option](https://www.sqlite.org/fts3.html#section_6_4)
    public enum MatchInfo: String {
        case fts3
    }

    /// [FTS4 options](https://www.sqlite.org/fts3.html#fts4_options)
    public enum Order: String {
        /// Data structures are optimized for returning results in ascending order by docid (default)
        case asc
        /// FTS4 stores its data in such a way as to optimize returning results in descending order by docid.
        case desc
    }

    var compressFunction: String?
    var uncompressFunction: String?
    var languageId: String?
    var matchInfo: MatchInfo?
    var order: Order?

    override public init() {
    }

    /// [The compress= and uncompress= options](https://www.sqlite.org/fts3.html#section_6_1)
    @discardableResult open func compress(_ functionName: String) -> Self {
        compressFunction = functionName
        return self
    }

    /// [The compress= and uncompress= options](https://www.sqlite.org/fts3.html#section_6_1)
    @discardableResult open func uncompress(_ functionName: String) -> Self {
        uncompressFunction = functionName
        return self
    }

    /// [The languageid= option](https://www.sqlite.org/fts3.html#section_6_3)
    @discardableResult open func languageId(_ columnName: String) -> Self {
        languageId = columnName
        return self
    }

    /// [The matchinfo= option](https://www.sqlite.org/fts3.html#section_6_4)
    @discardableResult open func matchInfo(_ matchInfo: MatchInfo) -> Self {
        self.matchInfo = matchInfo
        return self
    }

    /// [FTS4 options](https://www.sqlite.org/fts3.html#fts4_options)
    @discardableResult open func order(_ order: Order) -> Self {
        self.order = order
        return self
    }

    override func options() -> Options {
        var options = super.options()
        for (column, _) in (columnDefinitions.filter { $0.options.contains(.unindexed) }) {
            options.append("notindexed", value: column)
        }
        if let languageId {
            options.append("languageid", value: languageId)
        }
        if let compressFunction {
            options.append("compress", value: compressFunction)
        }
        if let uncompressFunction {
            options.append("uncompress", value: uncompressFunction)
        }
        if let matchInfo {
            options.append("matchinfo", value: matchInfo.rawValue)
        }
        if let order {
            options.append("order", value: order.rawValue)
        }
        return options
    }
}
