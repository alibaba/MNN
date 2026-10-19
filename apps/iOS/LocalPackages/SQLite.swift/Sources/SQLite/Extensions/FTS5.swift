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
    public static func FTS5(_ config: FTS5Config) -> Module {
        Module(name: "fts5", arguments: config.arguments())
    }
}

/// Configuration for the [FTS5](https://www.sqlite.org/fts5.html) extension.
///
/// **Note:** this is currently only applicable when using SQLite.swift together with a FTS5-enabled version
/// of SQLite.
open class FTS5Config: FTSConfig {
    public enum Detail: String {
        /// store rowid, column number, term offset
        case full
        /// store rowid, column number
        case column
        /// store rowid
        case none
    }

    var detail: Detail?
    var contentRowId: Expressible?
    var columnSize: Int?

    override public init() {
    }

    /// [External Content Tables](https://www.sqlite.org/fts5.html#section_4_4_2)
    @discardableResult open func contentRowId(_ column: Expressible) -> Self {
        contentRowId = column
        return self
    }

    /// [The Columnsize Option](https://www.sqlite.org/fts5.html#section_4_5)
    @discardableResult open func columnSize(_ size: Int) -> Self {
        columnSize = size
        return self
    }

    /// [The Detail Option](https://www.sqlite.org/fts5.html#section_4_6)
    @discardableResult open func detail(_ detail: Detail) -> Self {
        self.detail = detail
        return self
    }

    override func options() -> Options {
        var options = super.options()
        if let contentRowId {
            options.append("content_rowid", value: contentRowId)
        }
        if let columnSize {
            options.append("columnsize", value: Expression<Int>(value: columnSize))
        }
        if let detail {
            options.append("detail", value: detail.rawValue)
        }
        return options
    }

    override func formatColumnDefinitions() -> [Expressible] {
        columnDefinitions.map { definition in
            if definition.options.contains(.unindexed) {
                return " ".join([definition.0, Expression<Void>(literal: "UNINDEXED")])
            } else {
                return definition.0
            }
        }
    }
}
