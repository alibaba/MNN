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

/// A collating function used to compare to strings.
///
/// - SeeAlso: <https://www.sqlite.org/datatype3.html#collation>
public enum Collation {

    /// Compares string by raw data.
    case binary

    /// Like binary, but folds uppercase ASCII letters into their lowercase
    /// equivalents.
    case nocase

    /// Like binary, but strips trailing space.
    case rtrim

    /// A custom collating sequence identified by the given string, registered
    /// using `Database.create(collation:…)`
    case custom(String)

}

extension Collation: Expressible {

    public var expression: Expression<Void> {
        Expression(literal: description)
    }

}

extension Collation: CustomStringConvertible {

    public var description: String {
        switch self {
        case .binary:
            return "BINARY"
        case .nocase:
            return "NOCASE"
        case .rtrim:
            return "RTRIM"
        case .custom(let collation):
            return collation.quote()
        }
    }

}
