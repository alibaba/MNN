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

extension QueryType {

    /// Sets a `WITH` clause on the query.
    ///
    ///     let users = Table("users")
    ///     let id = Expression<String>("email")
    ///     let name = Expression<String?>("name")
    ///
    ///     let userNames = Table("user_names")
    ///     userCategories.with(userNames, as: users.select(name))
    ///     // WITH "user_names" as (SELECT "name" FROM "users") SELECT * FROM "user_names"
    ///
    /// - Parameters:
    ///
    ///   -  alias: A name to assign to the table expression.
    ///
    ///   -  recursive: Whether to evaluate the expression recursively.
    ///
    ///   -  hint: Provides a hint to the query planner for how the expression should be implemented.
    ///
    ///   -  subquery: A query that generates the rows for the table expression.
    ///
    /// - Returns: A query with the given `ORDER BY` clause applied.
    public func with(_ alias: Table, columns: [Expressible]? = nil, recursive: Bool = false,
                     hint: MaterializationHint? = nil, as subquery: QueryType) -> Self {
        var query = self
        let clause = WithClauses.Clause(alias: alias, columns: columns, hint: hint, query: subquery)
        query.clauses.with.recursive = query.clauses.with.recursive || recursive
        query.clauses.with.clauses.append(clause)
        return query
    }

    /// self.clauses.with transformed to an Expressible
    var withClause: Expressible? {
        guard !clauses.with.clauses.isEmpty else {
            return nil
        }

        let innerClauses = ", ".join(clauses.with.clauses.map { (clause) in
            let hintExpr: Expression<Void>?
            if let hint = clause.hint {
                hintExpr = Expression<Void>(literal: hint.rawValue)
            } else {
                hintExpr = nil
            }

            let columnExpr: Expression<Void>?
            if let columns = clause.columns {
                columnExpr = "".wrap(", ".join(columns))
            } else {
                columnExpr = nil
            }

            let expressions: [Expressible?] = [
                clause.alias.tableName(),
                columnExpr,
                Expression<Void>(literal: "AS"),
                hintExpr,
                "".wrap(clause.query) as Expression<Void>
            ]

            return " ".join(expressions.compactMap { $0 })
        })

        return " ".join([
            Expression<Void>(literal: clauses.with.recursive ? "WITH RECURSIVE" : "WITH"),
            innerClauses
        ])
    }
}

/// Materialization hints for `WITH` clause
public enum MaterializationHint: String {

    case materialized = "MATERIALIZED"

    case notMaterialized = "NOT MATERIALIZED"
}

struct WithClauses {
    struct Clause {
        var alias: Table
        var columns: [Expressible]?
        var hint: MaterializationHint?
        var query: QueryType
    }
    /// The `RECURSIVE` flag is applied to the entire `WITH` clause
    var recursive: Bool = false

    /// Each `WITH` clause may have multiple subclauses
    var clauses: [Clause] = []
}
